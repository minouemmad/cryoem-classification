
# CRYO-EM Classification Analysis


This program is designed to analyze particle classification data from CryoSPARC .cs files, producing various visualizations and statistical analyses to understand class assignments, probability distribution, and clustering performance.

The program has the following features:

* **normalized probability histograms**: normalized histograms of the probability each particle in the dataset is assigned to a given class
* **confusion matrix**: confusion matrix of true (row) vs predicted (column) class assignments, with each cell in the matrix containing mean and standard deviation information. For instance, a cell in the 2nd row and 3rd column can be interpreted as 'the probability a particle assigned to class 2 would be assigned to class 3.' 
* **shannon entropy histograms**: Shannon entropy distributions for each class
* **covariance scatterplots**: pairwise class assignment probability comparisons
* **k means clustering**: reclassification of particles using k means clustering
* **Gaussian Mixture Modeling**" reclassification of particles using Gaussian Mixture modeling


## Requirements

Download necessary Python libraries using the following code in terminal.  To run the cryo-EM particle classification software on a LINUX system, you will need:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

To run the PDB file processing software on Mac OSX, you will need:
'pip3 install pandas matplotlib biopython ramachandraw'


## Usage

The program needs a user input for the .cs file from CryoSPARC. This would preferably
be done using the full file path.

All plots will display automatically when the program is run EXCEPT for covariance plots. 
Covariance plots can be created by using the covariance_plot(class1, class2) function below.
