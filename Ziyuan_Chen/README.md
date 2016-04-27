#Income Predictor using 1995 Census Data compiled by UCI Machine Learning Lab
Data source: http://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)

This predictor uses seven features from the datasource to predict a person's income level.

The flow is:<br />
    1. Retrieve data from HDFS<br />
    2. Extract useful feature columns<br />
    3. Encode non-numerical features<br />
    4. Train the classifier<br />
    5. Predict using test data<br />
    6. Compute evaluation metrics<br />

The user can choose one of the following classifiers:<br />
    1. Decision Tree<br />
    2. Support Vector Machine (SVM)<br />
    3. k-Nearest-Neighbor (kNN), where k=15<br />

Several columns of the data source contain too many null entries and are therefore not used in the predication.<br />
