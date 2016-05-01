#Income Predictor using 1994 and 1995 Census Data Compiled by UCI Machine Learning Lab
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

The combined_predict() function is used to return the input matrix (50000-by-6) so that it can be aggregated with data from other sources in the combined prediction. To use it: <br />
	1. from predictor import * <br />
        2. p=prediction_engine('1','2'), the parameters do not matter in this case <br />
	3. x,y=p.combined_predict, where x is the matrix of features and y is a 1-D array containing the income
