# LAG-MNG
Collection of scripts, functions and libraries from Laser Analytics Group and Molecular Neuroscience Group of University of Cambridge

## Outline
The current purpose of this repository is to facilitate interactive classes on Python held within the group. The material for these classes are for now kept in the root of the repository. The classes include:
- Basic introduction to programming with Python 
- Modelling of epidemic growth with an ODE (based on COVID-19 data) and solving the equations using numerical integration
- Extracting features from images using a neural network in order to categorise images (`CNN-FeatureExtraction`)
- Denoising of synthetically corrupted bio-images using a convolutional neural network (`CNN-Denoising`)


## Data
We use the following datasets in some of the material:
- PatchCamelyon (https://github.com/basveeling/pcam)[PCam]; we only use the test set - download (https://drive.google.com/file/d/1hJ9MUaEPA90BRTnTHh2xWMWtyz6wWY-n/view?usp=sharing)[here]
- DIV2K (https://data.vision.ee.ethz.ch/cvl/DIV2K/)[DIVerse 2K]; we use a downsampled version of the test set - download (https://drive.google.com/drive/folders/1-DvfdLl3WUXVAYQSAgNjYo1v4HJSTslJ?usp=sharing)[here]

Make sure to put the datasets in the root folder of the cloned repository or change the corresponding paths accordingly in the Python code.
