# A Foundation Model Identifies Broad-Spectrum Antimicrobial Peptides against Drug-Resistant Bacterial Infections

# peptide language-based deep generative model (deepAMP)

This repository contains the code for the paper 'A Foundation Model Identifies Broad-Spectrum Antimicrobial Peptides against Drug-Resistant Bacterial Infections'.

## Dependencies

The following packages are required to run the code:

- python==3.8.0
- torch==1.12.1

To install the dependencies, run:

pip install -r requirements.txt

## deepAMP
├── code

│   ├── DataPath.py

│   ├── configer.py

│   ├── dataPre.py

│   ├── dataReady.py

│   ├── do_pack_sample.py

│   ├── do_sample.py

│   ├── do_train.py

│   └── rwHelper.py

├── data                             

│   ├── AMP_train_data.csv           # deepAMP-AOM original training dataset

│   ├── penetratin_train_data.csv    # deepAMP-POM original training dataset

│   └── uni_all_60.zip                # deepAMP-general training set

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
