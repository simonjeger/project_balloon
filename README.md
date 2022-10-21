# Project Balloon (February 2021 - April 2022)

This project is part of a semester- and master-thesis with the goal to get a balloon from a starting point to a pre-defined target point in 3D-space.<br />
<p align="center">
<img src="docs/outdoor_prototype.png" alt="outdoor prototype" width="300"/>
</p>


## Installation
### Create a new conda environment
```
cd project_balloon
conda create --name balloon --file requirements.txt
conda activate balloon
conda install pip
pip install descartes==1.1.0
pip install gym==0.22.0
```
Get [torch](https://pytorch.org/get-started/locally/) depending on your hardware

## Getting started
## Folder structure
- arduino
    - Was used to control the indoor prototype during the tests in London
- python3
    - Contains everything regarding learning
```
cd python3
```

## Training (requires downloading a training set)
Uncomment the line ```import agent_train``` in setup.py.<br />
It's recommended to generate multiple config files (see generate_yaml.py) and train in parallel.
```
python3 setup.py yaml/config_train.yaml
```

## Testing with pretrained model
The data provided in the data_example folder is a collection of eleven wind maps used during real life flight tests.<br />
```
python3 setup.py yaml/config_test.yaml
```

## Visualization
Can be turned on or off in config_NNNNN.py through the parameter ```render```.
- Top window: x-plane (side view)
- Middle window: y-plane (side view)
- Bottom window: z-plane (top-down view)
- White point: Target with acceptance radius
- Red point: balloon
- Red line: current action predicted by algorithm
- White line: path taken by balloon
- Grey line: random roll out used to set reachable target
- Orange background: wind blowing from left to right
- Blue background: wind blowing from right to left
<br/><br/>
<p align="center">
<img src="docs/render.png" alt="render" width="500"/>
</p>

## Advanced steps
### Generate config files for data acquisition or training.
Multiple files are generated to allow data acquisition or training in parallel (e.g. on a cluster). If you're running this on a standard computer we suggest only training with a few config files at the same time.<br />
If you are using a cluster, adapt and use the generated submit.txt file to submit jobs easier. The following command will store files in the ```yaml``` folder.
```
python3 generate_yaml.py
```

### Wind Data acquisition
There are two modes of wind-data acquisition. Set ```big_file``` parameter in convert_meteomatics.py accordingly.<br />
Either a single dataset with wind data of the current day is acquired, or a big file with data spanning over a month is downloaded which will then have to be further cut into peaces by build_set.py. <br />
Daily wind data set:
```
cd meteomatics
python3 meteomatics/convert_meteomatics.py ../yaml/config_NNNNN.py
```
For a big set over multiple months we suggest that you generate one config file for each month (parameter ```m``` in config file) and download them in parallel. <br />
In our experiments we used data from a full year (2021). Our training set consists of 1000 wind maps, the testing set of 300 wind maps.<br />:
```
cd meteomatics
python3 meteomatics/convert_meteomatics.py ../yaml/config_NNNN0.py
python3 meteomatics/convert_meteomatics.py ../yaml/config_NNNN1.py
...
python3 meteomatics/convert_meteomatics.py ../yaml/config_NNN12.py
python3 build_set.py ../yaml/config_NNNN0.py
```

### python3 generate_noise.py yaml/config_NNNNN.py
### Generate noise
This generates noise files (folder noise_MAPSIZE_XxMAPSIZE_Y) that are needed for training and testing.
```
python3 generate_noise.py
```