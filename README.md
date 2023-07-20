# ultralytics-env
Python base environment for model training using YOLOv8 by using Ultralytics module

## About
The main goal is to develop a base repo in which you can easily train a YOLOv8 model given a custom dataset. From here you can add any other feature _(like saving a video with the detections)_ the project might need, taking this repo as a base.

## Requirements
The training process is meant to be done in your GPU, using Cuda to be precise. The Python dependencies are listed in the `requirements.txt` file.

## Input
- **Pre-trained YOLOv8 model file path:** choose one depending on the [task](https://docs.ultralytics.com/tasks/) and download it from [Ultralytics website](https://docs.ultralytics.com/models/yolov8/#supported-modes:~:text=Training-,Performance,-Detection) and place it on `./models` folder.
- **Supervisely YOLOv8 image dataset:** download the dataset and place it on `./datasets` folder.
- **Output model file name:** type a name for the output model file.

## Output
- `<desired_name>.pt` trained model file with the best weights.
