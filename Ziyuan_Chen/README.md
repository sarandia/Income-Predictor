This predictor uses seven features from the datasource to predict a person's income level.

The flow is:
    1. Retrieve data from HDFS
    2. Extract useful feature columns
    3. Encode non-numerical features
    4. Train the classifier
    5. Predict using test data
    6. Compute evaluation metrics

Several columns of the data source contain too many null entries and are therefore not used in the predication.
