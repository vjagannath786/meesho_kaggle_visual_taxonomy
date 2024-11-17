# meesho_kaggle_visual_taxonomy

Steps to reproduce:
1) Download and store the data in data directory
2) Install requirements




For training & Inference:
With single GPU : python training.py
With multi GPU : torchrun --nproc_per_node=gpu_count training.py


For Inference only:
1) Make sure mered model is download from given link and placed in models folder
2) Run below command
python inference.py