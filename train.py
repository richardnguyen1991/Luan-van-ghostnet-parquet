import sys


if "--full-dataset" in sys.argv or "--stream-files" in sys.argv:
    from src.step8_full_train import main
else:
    from src.step4_train import main


if __name__ == "__main__":
    main()
