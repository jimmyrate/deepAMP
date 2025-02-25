# A Foundation Model Identifies Broad-Spectrum Antimicrobial Peptides against Drug-Resistant Bacterial Infections

# peptide language-based deep generative model (deepAMP)

This repository contains the code for the paper 'A Foundation Model Identifies Broad-Spectrum Antimicrobial Peptides against Drug-Resistant Bacterial Infections'.

## Dependencies

The following packages are required to run the code:

- python==3.8.0
- torch==1.12.1

To install the dependencies, run:

conda env create -f environment.yml

## deepAMP
├── code

│   ├── DataPath.py

│   ├── configer.py

│   ├── dataPre.py         # Parameter setting and model selection

│   ├── dataReady.py

│   ├── do_pack_sample.py

│   ├── do_sample.py     # Model Optimization Generation

│   ├── do_train.py      # Model training

│   ├── count_fitness_value.py  # count fitness value

│   └── rwHelper.py

├── data                             

│   ├── AOM-training-pairs.txt           # deepAMP-AOM training dataset

│   ├── POM-training-pairs.txt           # deepAMP-POM original training dataset

│   └── uni_all_60.zip                   # deepAMP-general training set

├── src                               

│   ├── models                        # Model definitions

│   ├── DataProcessTools.py

│   ├── utils.py

│   └── warmupScheduler.py

│ 

├── requirements.txt

└── README.md                         # This README file

## Trained model
The trained model is uploaded to https://drive.google.com/drive/folders/1rtlc8DcBeV6yOxmlXIwjoT6VXuYgaIFg?usp=drive_link


## Usage
Select the dataset for training and the parameters in code/dataPre.py, and then train the model using code/do_train.py

## Pretraining
1. select dataset
   In code/dataPre.py set the parameter: dataset='uniport_all_60'
2. run code/do_train.py

## Finetuning-AOM
1. select dataset
   In code/dataPre.py set the parameter: dataset='random_pair'
2. run code/do_train.py

## Finetuning-POM
1. select dataset
   In code/dataPre.py set the parameter: dataset='random_chem_penetratin_pair'
2. run code/do_train.py

## Reference

@article{li2024foundation,
  title={A foundation model identifies broad-spectrum antimicrobial peptides against drug-resistant bacterial infection},
  author={Li, Tingting and Ren, Xuanbai and Luo, Xiaoli and Wang, Zhuole and Li, Zhenlu and Luo, Xiaoyan and Shen, Jun and Li, Yun and Yuan, Dan and Nussinov, Ruth and others},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={7538},
  year={2024},
  publisher={Nature Publishing Group UK London}
}

