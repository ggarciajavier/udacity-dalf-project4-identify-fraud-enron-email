## Udacity Data Analyst Nanodegree Project 4: Identify fraud from Enron email ##

This repository contains the analysis performed by Javier Gómez García for the 4th project of the Udacity Data Analyst Nanodegree.
It includes the following files:

* A Jupyter notebook containing the main program and the description of the work (Python version: 2.7):
    [Project4_Identify_Fraud_from_Enron_Email.ipynb](Project4_Identify_Fraud_from_Enron_Email.ipynb)

* The following text files:
    * [REFERENCES](REFERENCES.md): text file containing the links to the websites and repositories I have used for my work.

* The following Python files, required to process the data and called by the main program (the .ipynb file), all written in Python 2.7:
    * [poi_id.py](poi_id.py): main program to build the POI identification algorithm
    * [helper_functions.py](helper_functions.py): additional functions required for processing the data and doing the tuning and testing of the classifiers
    * [tester.py](tester.py): additional functions to test the classifier and write the output pickle files (provided by Udacity)
    * [feature_format.py](feature_format.py): additional functions to process the data (provided by Udacity)
	
* The following pickle files (.pkl):
    * [final_project_dataset.pkl](final_project_dataset.pkl): input file containing the Enron dataset (provided by Udacity)
    * [my_classifier.pkl](my_classifier.pkl): file containing the final classifier
    * [my_dataset.pkl](my_dataset.pkl): file containing the dataset with the new features added
    * [my_feature_list.pkl](my_feature_list.pkl): file containing the names of the features used for the classifier