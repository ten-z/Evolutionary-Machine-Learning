# VUW Glacier Data Prediction

This repository contains the implementation and documentation for the VUW Glacier Data Prediction project. Its primary aim is to forecast glacier‐related variables (such as ice thickness) using machine learning techniques. The core functionality includes:

- **Data Restructuring:**  
  Reading, cleaning, and reorganizing raw glacier/ice field data into a consistent tabular format suitable for modeling.

- **Feature Selection:**  
  Applying a Genetic Algorithm (GA) to identify the most informative feature subsets before training predictive models.

- **Model Training & Evaluation:**  
  Training a neural network (MLPRegressor) on the selected features and evaluating its performance on held‐out data.
