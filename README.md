Project Balloon (February 2021 - April 2022)

This project is part of a semester- and master-thesis with the goal to get a balloon from a starting point to a pre-defined target point in 3D-space.
To use this code, the following steps should be done:

generate_yaml.py
This generates yaml files (config_NNNNN.py) that are needed for training and testing. All the important parameters can be set there.

meteomatics/convert_meteomatics.py ../yaml/config_NNNNN.py
This downloads either a big set of data (big_file = True) that then needs to be converted into a set using build_set.py or generates a dataset of wind data on the current day (big_file = False)

setup.py yaml/config_NNNNN.py
This generates the folder structure and trains the algorithm with a short testing at the end

agent_test.py yaml/config_NNNNN.py
This tests the trained weights
