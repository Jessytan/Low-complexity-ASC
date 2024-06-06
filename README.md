# Low-complexity-ASC
This repository contains the introductions to the datasets and code used in our paper, titled "Low-Complexity Acoustic Scene Classification Using Parallel Attention-Convolution Network" (as shown in the section of Citation).
- [Datasets](#datasets)
- [Enviroment](#environment)
- [Code](#code)
	- [reuse](#reuse)
	- [main](#main)
# datasets
We conduct our experiments on the TAU Urban Acoustic Scene 2022 Mobile development dataset (TAU22) which consists of audio clips acquired by mobile devices in urban environments. You can download from [here](https://doi.org/10.5281/zenodo.6337421).
# enviroment
The required library files are placed in requirements.txt  
Our environment: RTX3090 + cuda11.3 + torch1.11
# code
## reuse
You need to create a directory named reuse to save training and validation data, and in the [estimate_devices_freq.ipynb](estimate_devices_freq.ipynb) will used for spectrum modulation.
## main
[student.ipynb](student.ipynb) is the main code for training and validating the datasets.  
Teacher models are same with the submission to the DCASE2023, and some are pretrained and provided in [teacher_models](teacher_models).

