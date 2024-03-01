# Low-complexity-ASC
This repository is dedicated to the submission to IEEE ICASSP2024, and the Tan submission of DCASE2023,Task 1, Low-Complexity Acoustic Scene Classification.
- [Datasets](#datasets)
- [Enviroment](#environment)
- [Code](#code)
	- [reuse](#reuse)
	- [main](#main)
 - [Contact](contact)
# datasets
We conduct our experiments on the TAU Urban Acoustic Scene 2022 Mobile development dataset (TAU22) which consists of audio clips acquired by mobile devices in urban environments. You can download from [here](https://doi.org/10.5281/zenodo.6337421).
# enviroment
The required library files are placed in requirements.txt  
Our environment: RTX3090 + cuda11.3 + torch1.11
# code
## reuse
You need to create a directory named reuse to save training and validation data, and in the [estimate_devices_freq.ipynb](estimate_devices_freq.ipynb) will used for spectrum modulation.
## main
[student.ipynb](main.ipynb) is the main code for training and validating the datasets.  
Teacher models are same with the submission to the DCASE2023, and some are pretrained and provided in [teacher_models](teacher_models).

