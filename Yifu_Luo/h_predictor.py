import numpy
import pandas
import math
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

class h_predictor:
    def __init__(self):
        # matrix containing the data
        self.matrix = []

        # number of datapoints (i.e persons)
        self.num_of_datapoints = 500000

        # classification range of salary
        # e.g. if this number is 25000, the levels will be:
        # 0 = [0 .. 24999], 1 = [25000 .. 49999], 2 = [50000 .. 74999], etc
        self.classification_range = 30000

        # portion of the testing data in relative to total data
        # 0.4 means 40% testing data and 60% training data
        self.test_portion = 0.25

        # the list of features of interest to use
        # [Record Type (RT), Housing serial number (SERIALNO), State (ST), Age (AGEP), 
        # Citizenship Status (CIT), Mean of transportation to work (JWTR),
        # Marital status (MAR), Educational attainment (SCHL), Sex (SEX), Wages or Salary income 
        # past 12 months (WAGP), Usual hours worked per week past 12 months (WKHP), WKW (weeks 
        # work during last year), Recoded field of degree (FOD1P), Industry Code (INDP)]
        self.all_features_list = ['RT', 'SERIALNO', 'ST', 'AGEP', 'CIT', 'JWTR', 'MAR', 'SCHL', 'SEX', \
                                  'WAGP', 'WKHP', 'WKW', 'FOD1P', 'INDP']

        # useful features that could be use and single feature in prediction
        # list containing the index of features for prediction:
        # 0 = RT, 1 = SERIALNO, 2 = ST, 3 = AGEP, 4 = CIT, 5 = JWTR, 6 = MAR, 7 = SCHL, 8 = SEX,
        # 10 = WKHP, 11 = WKW, 12 = FOD1P, 13 = INDP
        # The list should not contains 9 because salary is target that we are trying to predict
        self.useful_features = [3, 4, 5, 6, 7, 8, 10, 11, 12, 13]


    # load the data from file into numpy array for group combined project
    def load_data(self, filename):
        # Read data from file
        # Use self.num_of_datapoints if it is set, else load all the file
        if self.num_of_datapoints is None:
            dataframe = pandas.read_csv(filename, na_values='', header=0, usecols=self.all_features_list)
        else:
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


    # format the matrix data into target (salary) and features numpy arrays
    # this function needs to be recalled anytime the features
    # of the prediction change to avoid null or nan data
    # feature_indexes = the indexes of values that will be used as features in our prediction,
    # taken from the self.useful_features array
    # is_classification = boolean value determines if the test is classification into salary
    # ranges vs. regression of salary
    def format_data(self, feature_indexes, is_classification=True):
        salary = []
        features = []
        for person in self.matrix:
            if person[0] == 'P':
                wage = person[9]
                if not math.isnan(wage):
                    tmp_datapoint = []
                    for index in feature_indexes:
                        try:
                            person[index]
                        except IndexError:
                            print 'error feature not present'
                        value = person[index]
                        if value == 'NaN' or math.isnan(value):
                            value = 0
                        tmp_datapoint.append(float(value))

                    # append the salary and features to respective list
                    if is_classification:
                        salary.append(int(round(wage / self.classification_range)))
                    else:
                        salary.append(wage)
                    features.append(tmp_datapoint)
                
        '''
        # test for data formatting
        for i in range(0, len(salary)):
            print salary[i], 'and', features[i]
        '''
        return salary, features


    # Returns a salary numpy array and a features numpy array for team work combination
    # The format of the features array is: 
    # [AGE, SEX, EDUCATION, INDUSTRY_CODE, WEEKS_WORKED_LAST_YEAR]
    # which corresponds to our feature indexes: [3, 8, 7, 13, 11]
    # There are also important encoding fixtures in this function to make sure
    # the values of the features are compatible to group's data
    def format_data_combine(self, filename, feature_indexes=[3, 8, 7, 13, 11]):
        self.num_of_datapoints = 50000
        self.load_data(filename)
        salary = []
        features = []
        for person in self.matrix:
            if person[0] == 'P':
                wage = person[9]
                if not math.isnan(wage):
                    tmp_datapoint = []
                    # determines if the datapoint is valid so that it would be
                    # appende to the matrix
                    append = True
                    for index in feature_indexes:
                        try:
                            person[index]
                        except IndexError:
                            print 'error feature not present'
                        value = person[index]
                        if not value == 'NaN' and not math.isnan(value):
                            # AGE (index == 3), INDUSTRY_CODE (index == 13)
                            # and WEEKS_WORKED_LAST_YEAR feature always pass
                            # SEX feature
                            if index == 8:
                                # my data has the Male as 1 and Female as 2
                                # group data has Male as 0 and Female as 1
                                value = value - 1
                            
                            # Education Level SCHL feature
                            elif index == 7:
                                if value in [4, 5, 6, 7]:
                                    value = 4
                                elif value in [8, 9, 10, 11]:
                                    value = 5
                                elif value in [12, 13, 14, 15]:
                                    value -= 6
                                elif value in [16, 17]:
                                    value = 10
                                elif value in [18, 19]:
                                    value = 11
                                elif value == 20:
                                    value = 13
                                elif value in [21, 22, 23, 24]:
                                    value -= 7
                            tmp_datapoint.append(float(value))

                        else:
                             append = False
                        
                    # append the salary and features to respective list
                    if append:
                        if wage > 50000:
                            salary.append(1)
                        else:
                            salary.append(0)
                        features.append(tmp_datapoint)
        '''        
        # test for data formatting
        for i in range(0, len(salary)):
            print salary[i], 'and', features[i]
        '''
        from numpy import array
        return array(features), array(salary)

    
    # train and predict the dataset
    def run(self):        
        # this portion contains classification test with single feature
        print 'Single feature predictions:'
        print 'Starting classification test...'
        for i in self.useful_features:
            self.classification([i])

        # this portion contains regression test with single feature
        print "Starting regression test..."
        for i in self.useful_features:
            self.regression([i])

        print '\nTwo features predictions:'
        # this portion contains classification on two features:
        print 'Starting classification test...'
        self.classification([10, 11])
        self.classification([10, 12])
        self.classification([7, 10])
        self.classification([5, 10])

        # this portion contains regression on two features:
        self.regression([10, 11])
        self.regression([10, 12])
        self.regression([7, 10])
        self.regression([5, 10])

        print '\nAll features predictions:'
        # this portion contains classifcation on all features
        print 'Starting classification test...'
        self.classification(self.useful_features)

        # this portion contains regression on all features
        print "Starting regression test..."
        self.regression(self.useful_features)
        

    # classification test on list of features
    # features_idx_array should contains the indexes of the features that we want to predict on.
    # The indexes come from: 3 = AGEP, 4 = CIT, 5 = JWTR, 6 = MAR, 7 = SCHL, 8 = SEX,
    # 10 = WKHP, 11 = WKW, 12 = FOD1P, 13 = INDP
    # For example, features_idx_array = [7, 10] means we want [SCHL, WKHP] as features
    def classification(self, features_idx_array):
        salary, features = self.format_data(features_idx_array, True)
        
        # randomize the datapoints to avoid bias, also split the data into training and testing group
        feature_train, feature_test, target_train, target_test = train_test_split(features, salary, \
                                                                                  test_size=self.test_portion, \
                                                                                  random_state=42)
        # list of classification classifiers used in this prediction
        # more classifier could be added into both names and clfs
        names = ['Naive Bayes', 'Decision Tree']
        clfs = [
            MultinomialNB(),
            DecisionTreeClassifier(min_samples_split=4, max_depth=6)]
        features_names = [self.all_features_list[i] for i in features_idx_array]
        for name, clf in zip(names, clfs):
            clf.fit(feature_train, target_train)
            pred = clf.predict(feature_test)
            print 'Accuracy score for feature ',features_names,' with ', \
                name, ': ', accuracy_score(pred, target_test)
        print '\n'
 

    # regression test on list of features
    def regression(self, features_idx_array):
        salary, features = self.format_data(features_idx_array, False)
        
        feature_train, feature_test, target_train, target_test = train_test_split(features, salary,\
                                                                                  test_size=self.test_portion,\
                                                                                  random_state=42)
        reg = LinearRegression()
        reg.fit(feature_train, target_train)
        features_names = [self.all_features_list[i] for i in features_idx_array]
        pred = reg.predict(feature_test)
        print 'R^2 score for feature ',features_names,' with regression', reg.score(feature_test, target_test)
        

if __name__ == '__main__':
    p = predictor()
    # features, labels = p.format_data_combine('ss13pusb.csv')
    # print type(features), type(labels)
    p.load_data('ss13pusb.csv')
    p.run()
