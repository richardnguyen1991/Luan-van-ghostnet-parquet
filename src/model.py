from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class GraphBatch:
    sequence_x: torch.Tensor
    target_y: torch.Tensor
    edge_index: tuple[torch.Tensor, ...]
    node_counts: torch.Tensor

    def to(self, device: torch.device | str) -> "GraphBatch":
        return GraphBatch(
            sequence_x=self.sequence_x.to(device),
            target_y=self.target_y.to(device),
            edge_index=tuple(edge.to(device) for edge in self.edge_index),
            node_counts=self.node_counts.to(device),
        )


class DirectedGraphConv(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(dimension, dimension, bias=False)
        self.in_projection = nn.Linear(dimension, dimension, bias=False)
        self.out_projection = nn.Linear(dimension, dimension, bias=False)
        self.normalization = nn.LayerNorm(dimension)
        self.activation = nn.ReLU()

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edge_count]")
        source, destination = edge_index
        incoming = torch.zeros_like(nodes)
        outgoing = torch.zeros_like(nodes)
        incoming_degree = torch.zeros(nodes.shape[0], device=nodes.device, dtype=nodes.dtype)
        outgoing_degree = torch.zeros_like(incoming_degree)
        if source.numel():
            incoming.index_add_(0, destination, nodes[source])
            outgoing.index_add_(0, source, nodes[destination])
            incoming_degree.index_add_(0, destination, torch.ones_like(destination, dtype=nodes.dtype))
            outgoing_degree.index_add_(0, source, torch.ones_like(source, dtype=nodes.dtype))
        incoming = incoming / incoming_degree.clamp_min(1.0).unsqueeze(1)
        outgoing = outgoing / outgoing_degree.clamp_min(1.0).unsqueeze(1)
        update = (
            self.self_projection(nodes)
            + self.in_projection(incoming)
            + self.out_projection(outgoing)
        )
        return self.activation(self.normalization(update))


class GhostModule1d(nn.Module):
    def __init__(self, input_channels: int, primary_channels: int, ratio: int, dropout: float) -> None:
        super().__init__()
        cheap_channels = primary_channels * max(1, int(ratio) - 1)
        self.primary = nn.Sequential(
            nn.Conv1d(input_channels, primary_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(primary_channels),
            nn.ReLU(),
        )
        self.cheap = nn.Sequential(
            nn.Conv1d(
                primary_channels,
                cheap_channels,
                kernel_size=3,
                padding=1,
                groups=primary_channels,
                bias=False,
            ),
            nn.BatchNorm1d(cheap_channels),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.output_channels = primary_channels + cheap_channels

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        primary = self.primary(values)
        return self.dropout(torch.cat((primary, self.cheap(primary)), dim=1))


class GCLSTMGhostNet(nn.Module):
    def __init__(self, feature_count: int, class_count: int, config: dict[str, Any]) -> None:
        super().__init__()
        step4 = config["step4"]
        flow_dim = int(step4["flow_embedding_dim"])
        graph_dim = int(step4["graph_hidden_dim"])
        self.flow_encoder = nn.Sequential(
            nn.Linear(int(feature_count), flow_dim),
            nn.LayerNorm(flow_dim),
            nn.ReLU(),
        )
        self.node_input = nn.Linear(flow_dim, graph_dim)
        self.graph_layers = nn.ModuleList(
            DirectedGraphConv(graph_dim) for _ in range(int(step4["graph_layers"]))
        )
        concatenated_graph_dim = graph_dim * int(step4["graph_layers"])
        self.spatial_attention = nn.Linear(concatenated_graph_dim, 1)
        self.edge_fusion = nn.Sequential(
            nn.Linear(flow_dim + concatenated_graph_dim * 2, graph_dim),
            nn.LayerNorm(graph_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=graph_dim,
            hidden_size=int(step4["lstm_hidden_dim"]),
            num_layers=int(step4["lstm_layers"]),
            batch_first=True,
            dropout=float(step4["dropout"]) if int(step4["lstm_layers"]) > 1 else 0.0,
            bidirectional=bool(step4["bidirectional_lstm"]),
        )
        lstm_output = int(step4["lstm_hidden_dim"]) * (2 if step4["bidirectional_lstm"] else 1)
        self.temporal_attention = nn.Linear(lstm_output, 1)
        self.ghost = GhostModule1d(
            lstm_output,
            int(step4["ghost_primary_channels"]),
            int(step4["ghost_ratio"]),
            float(step4["dropout"]),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.ghost.output_channels + lstm_output, lstm_output),
            nn.ReLU(),
            nn.Dropout(float(step4["dropout"])),
            nn.Linear(lstm_output, int(class_count)),
        )

    def _spatial_sequence(
        self,
        flow_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        node_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, destination = edge_index
        if edge_index.shape[1] != flow_embeddings.shape[0]:
            raise ValueError("Each temporal flow requires exactly one graph edge")
        nodes = flow_embeddings.new_zeros((int(node_count), flow_embeddings.shape[1]))
        degree = flow_embeddings.new_zeros((int(node_count),))
        nodes.index_add_(0, source, flow_embeddings)
        nodes.index_add_(0, destination, flow_embeddings)
        degree.index_add_(0, source, torch.ones_like(source, dtype=flow_embeddings.dtype))
        degree.index_add_(0, destination, torch.ones_like(destination, dtype=flow_embeddings.dtype))
        nodes = self.node_input(nodes / degree.clamp_min(1.0).unsqueeze(1))
        layer_outputs = []
        for layer in self.graph_layers:
            nodes = layer(nodes, edge_index)
            layer_outputs.append(nodes)
        nodes = torch.cat(layer_outputs, dim=1)
        spatial_weights = torch.softmax(self.spatial_attention(nodes).squeeze(-1), dim=0)
        attended_nodes = nodes * spatial_weights.unsqueeze(1)
        fused = self.edge_fusion(
            torch.cat((flow_embeddings, attended_nodes[source], attended_nodes[destination]), dim=1)
        )
        return fused, spatial_weights

    def forward(self, batch: GraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        flow = self.flow_encoder(batch.sequence_x)
        spatial_outputs = [
            self._spatial_sequence(flow[index], batch.edge_index[index], int(batch.node_counts[index]))
            for index in range(flow.shape[0])
        ]
        spatial = torch.stack([item[0] for item in spatial_outputs])
        self.last_spatial_attention = tuple(item[1] for item in spatial_outputs)
        temporal, _ = self.lstm(spatial)
        attention = torch.softmax(self.temporal_attention(temporal).squeeze(-1), dim=1)
        attended = torch.sum(temporal * attention.unsqueeze(-1), dim=1)
        ghost_features = self.ghost(temporal.transpose(1, 2)).mean(dim=2)
        return self.classifier(torch.cat((attended, ghost_features), dim=1)), attention


def model_parameter_count(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
