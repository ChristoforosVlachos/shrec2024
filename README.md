# Shrec 2024: Recognition of hand motions molding clay

This repository contains a submission to the "Recognition of hand motions molding clay" track of the Shape Retrieval Contest (SHREC) 2024.

## Description (from [shrec.net](https://www.shrec.net/SHREC-2024-hand-motion/))
This contest will focus on trying to recognize different highly similar hand motions from a professional potter. Participants will try to recognize the hand motions from coordinate data of both potters hands. This track is made to see if hand recognition systems can recognize similar hand motions of two hands. These motions are all recorded using a Vicon system and pre-processed into a coordinate system using Blender and Vicon shogun Post. 

## Dataset (from [shrec.net](https://www.shrec.net/SHREC-2024-hand-motion/))
We recorded hand motions of an experienced potter who sculpted the same pot with and without clay. We captured our data using a Vicon System containing 14 Vantage Cameras that will track reflective markers on the subject's hands. The motions are processed to be saved as a text file where each row represents the data of a specific frame with 28 coordinate floats (14 per hand) (x;y;z positions) of the markers. The dataset is split into a training and testing set (70/30). 

The motion classes used in this task are:

- **Centering** the clay.
- **Making a hole** in the clay.
- **Pressing** the clay to make it stick to the pottery wheel.
- **Raising** the base structure of the clay.
- **Smoothing** the walls.
- Using the **sponge** to make the clay more moist.
- **Tightening** the cylinder of the clay.

The ground truth is manually added based on the motions we have captured. The motion class can be found in the filename and folder.

[Data split (Train/Test)](https://www.shrec.net/SHREC-2024-hand-motion/Data/Data%20Split.rar) for the data split in a train and test set.

## Instructions
This is a short description of how you may use the executable code for training/inference.

### Training
Run:

```shell
python training.py data_path n_epochs
```

This will train the network on the Dataset under `data_path` for `n_epochs` epochs. The Dataset must be formatted as follows:
```
data_path
    |-- Train-set
            |-- Class_1
                   |-- 01.txt
                   |-- 02.txt
                  ...
            |-- Class_2
                   |-- 01.txt
                   |-- 02.txt
                  ...
           ...
    |-- Test-set
            |-- Class_1
                   |-- 01.txt
                   |-- 02.txt
                  ...
            |-- Class_2
                   |-- 01.txt
                   |-- 02.txt
                  ...
           ...
```

Running `training.py` will create a directory `checkpoints` inside the current working directory and save the model's state dictionary for each epoch in the format `epoch_i.pt`. You will need to keep one of these files for inference.

PLEASE NOTE: `training.py` is dependent on `tqdm` being installed.

*Usage example:*
```shell
python training.py "Data Split" 1000
```

### Inference
Run:

```shell
python inference.py data_path model_file
```

This will run inference on every `.txt` file found under `data_path` in alphabetical order. The model's parameters will be set using the state dictionary `model_file`. As a `model_file` you may use the provided `model.pt` or a checkpoint from running `training.py` yourself.

After finishing the inference, a report will be generated in a file `Results.txt` inside the current working directory and will be simultaneously printed on the console. The format of the report is:
```
index (space) MotionClass
```

*Usage exapmple:*
```shell
python inference.py "Data Split/Test-set" model.pt
```

*Report example:*
```
1 Centering
2 Centering
3 Centering
4 MakingHole
5 Pressing
6 Pressing
7 Raising
8 Raising
9 Smoothing
10 Smoothing
11 Sponge
12 Tightening
```