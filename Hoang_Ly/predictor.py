import numpy
import pandas
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

class predictor:

    def __init__(self):
        # matrix containing the data
        self.matrix = []

        # number of datapoints (i.e persons)
        self.num_of_datapoints = 10000

        # the list of features of interest to use
        # [Record Type (RT), Housing serial number (SERIALNO), State (ST), Age (AGEP), 
        # Citizenship Status (CIT), Class of worker (COW), Mean of transportation to work (JWTR),
        # Marital status (MAR), Educational attainment (SCHL), Sex (SEX), Wages or Salary income 
        # past 12 months (WAGP), Usual hours worked per week past 12 months (WKHP), WKW (weeks 
        # work during last year), Recoded field of degree (FOD1P)]
        self.all_features_list = ['RT', 'SERIALNO', 'ST', 'AGEP', 'CIT', 'JWTR', 'MAR', 'SCHL', 'SEX', \
                    'WAGP', 'WKHP', 'WKW', 'FOD1P']

        # list containing the index of features for prediction:
        # 0 = RT, 1 = SERIALNO, 2 = ST, 3 = AGEP, 4 = CIT, 5 = JWTR, 6 = MAR, 7 = SCHL, 8 = SEX,
        # 10 = WKHP, 11 = WKW, 12 = FOD1P
        # The list should not contains 9 because salary is target that we are trying to predict
        self.feature_indexes = [7]

    # load the data from file into numpy array
    def load_data(self, filename):
        
        # Read data from file
        dataframe = pandas.read_csv(filename, na_values='', header=0, \
                                    usecols=self.all_features_list, nrows=self.num_of_datapoints)

        self.matrix = dataframe.values

    # test the data loaded from file
    def test_data(self):
        data_list = self.matrix.tolist()

        for person in data_list:
            print person

    # return the matrix data
    def get_table(self):
        return self.matrix

    # get the features from the user
    def input_features(self):
        print 'Please enter feature(s) you would like to use:'

    # format the data into target and features
    def format_data(self):
        salary = []
        features = []
        for person in self.matrix:
            if person[0] == 'P':
                wage = person[9]
                if not math.isnan(wage):
                    tmp_datapoint = []
                    for index in self.feature_indexes:
                        try:
                            person[index]
                        except IndexError:
                            print 'error feature not present'
                        value = person[index]
                        if value == 'NaN':
                            value = 0
                        tmp_datapoint.append(float(value))

                    # append the salary and features to respective list
                    salary.append(int(round(wage / 20000)))
                    features.append(tmp_datapoint)

        return salary, features
                
    
    # train the dataset
    def train(self):
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        
        from sklearn.cross_validation import train_test_split
        salary, features = self.format_data()

        '''
        # test for data formatting
        for i in range(0, len(salary)):
            print salary[i], 'and', features[i]
        '''
        feature_train, feature_test, target_train, target_test = train_test_split(features, salary, test_size=0.7,\
                                                                                  random_state=42)

        clf.fit(feature_train, target_train)
        pred = clf.predict(feature_test)
        print accuracy_score(pred, target_test)

if __name__ == '__main__':
    p = predictor()
    p.load_data('ss13pus.csv')
#    p.test_data()
    p.train()
