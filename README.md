# BrainAgingTDA

This repository supplements the paper *Topological Detection of Alzheimer’s Disease using Betti Curves* by Ameer Saadat-Yazdi, Rayna Andreeva and Rik Sarkar. The study seeks to explore the relationship between structural MRI topology and Aging/Alzheimer's.

## Running experiments
### Downloading data
To run the experiments, the OASIS3 FreeSurfer data must be downloaded. See https://www.oasis-brains.org/ and https://github.com/NrgXnat/oasis-scripts for more details on how to do this. The downloaded data must be stored in the `./oasis` subdirectory. In addition to this, the MR Data, Freesurfers and ADRC Clinical Data tables must be downloaded as csv files from the OASIS XNAT server. They should be named `Oasis_MRI.csv`, `Oasis_FS.csv` and `Oasis_ClinicalData.csv` respectively and stored in the `./Data` directory.

### Preparing data
To prepare the data and compute the Betti curves first run `python data-prep.py` followed by `python compute-curves.py`. The latter script may take several hours to complete due to the complexity of topology computations.

### Training and evaluating models
Finally, you should now be able to run the **Experiments** notebook to train and evaluate the models used in the paper. Note that you will have to alter the code in order to train on different feature sets e.g. (gray matter vs white matter).
