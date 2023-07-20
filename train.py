import argparse

from pathlib import Path
from typing import Tuple

from ultralytics import YOLO

# TODO: Delete unused constants
BASE_MODEL = r"models/yolov8m-seg.pt"
CONF_FILE_PATH = r"./datasets/254437_CHUTE_3/data_config.yaml"


def args_parser():
    parser = argparse.ArgumentParser(
        description="Python script for YOLOv8 model training using a custom dataset"
    )
    parser.add_argument(
        "-m", "--model", type=str, help="YOLOv8 model file path", required=True
    )
    parser.add_argument(
        "-d", "--dataset", type=str, help="Dataset folder path", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Name of the ouput trained-model .pt file",
        required=True,
    )

    args = parser.parse_args()

    return args


def io(
    model_file_path_str: str, dataset_folder_path_str: str, output_model_name: str
) -> Tuple[str, str]:
    model_file_path = Path(model_file_path_str)
    if not model_file_path.exists():
        raise FileNotFoundError(
            "Model file does not exist. Check if the path is correct."
        )

    dataset_folder_path = Path(dataset_folder_path_str)
    if not dataset_folder_path.is_dir():
        raise FileNotFoundError(
            "Dataset folder does not exist. Check if the path is correct."
        )

    return model_file_path, dataset_folder_path, output_model_name


def main():
    args = args_parser()
    model_file_path, dataset_folder_path, output_model_name = io(
        args.model, args.dataset, args.output
    )

    model = YOLO(model_file_path)
    model.train(data=dataset_folder_path, epochs=100)

    # TODO: Track new folder created by Ultralytics and get the "best.pt" file and rename it to "output_model_name" value.

    # model = YOLO(BASE_MODEL)
    # model.train(data=CONF_FILE_PATH, epochs=100)


if __name__ == "__main__":
    main()
