# Depth Ordering for Comics
The code for my semester Master project "Depth Ordering for Comics".

As I worked on pre-existing models, they are for the most part take directly from the original projects. The README of the projects have been kept as they contain instructions to run the code, as well as some important links for the project. They were not updated for this project.

# AdaBins
Taken from https://github.com/shariqfarooq123/AdaBins

Modifications:
* Removed all files exclusive to the datasets of the original work
* Removed commented or redundant lines of code
* Added 'pascal' option in the infer.py (lines 86-93) file
* Added the necessary code to scale inputs in infer.py (lines 74-75, 82-83, 153-160 and 172-173) to make them compatible with the model, and to scale back the result to the original size
* Added the necessary code to scale inputs in dataloader.py (lines 93-103 and 159-162) to make them compatible with the model
* Added the necessary code to keep only the depth informations in depth maps in dataloader.py (lines 122-124 and 170-172)
* Created args_train_pascal.txt, containing the training informations for the model using the Pascal dataset

# SDFA-Net
Taken from https://github.com/ZM-Zhou/SDFA-Net_pytorch

Modifications:
* Removed all files exclusive to the datasets of the original work
* Fixed an incorrect function name (maybe because of pytorch version) in datasets/utils/my_transforms.py (line 39)
* Added the necessary code to load the UASOL dataset in \_\_init\_\_.py (line 8) and by writing datasets/uasol_dataset.py, based on the pre-existing (now removed) data loaders
* Created the option files in options/_base/datasets/uasol and the files options/SDFA-Net/train/uasol_train_stage1.yaml and options/SDFA-Net/train/uasol_train_stage2-yaml to make the UASOL dataset usable with the model
* Added the training command for UASOL in train_scripts.sh (lines 1-20)
* Removed the scaling in predict.py to have depth maps the same shape as the input image
* Changed all color maps in visualizer.py to 'inferno'
* Removed some options in path_my.py and \_\_init\_\_.py tied to the dataset used in the original work

# Evaluation
The evaluation code is based on the evaluation.py file from https://github.com/martin-ev/Estimating-image-depths-in-new-domains/tree/76b01d35a04d571c252abea9100d89302f76fb41#d-evaluating-the-models, liberaly changed to be cleaner and match what was needed for this project
