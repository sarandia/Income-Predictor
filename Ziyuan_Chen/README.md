This predictor uses seven features from the datasource to predict a person's income level.

The flow is:
    1. Retrieve data from HDFS
    2. Extract useful feature columns
    3. Encode non-numerical features
    4. Train the classifier
    5. Predict using test data
    6. Compute evaluation metrics

The user can choose one of the following classifiers:
    1. Decision Tree
    2. Support Vector Machine (SVM)
    3. k-Nearest-Neighbor (kNN), where k=15

Several columns of the data source contain too many null entries and are therefore not used in the predication.
