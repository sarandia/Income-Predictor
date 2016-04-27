import numpy
import scipy
import sklearn
import pandas
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
    
# Class predictor
# Used to analyze and perform machine learning tasks on the 1990 census data
# obtained from the UCI Machine Learning Repository
 
class predictor:
    def __init__(self, data_src, header_src):
        
        # Read in the parsed header file and construct a pandas table
        with open(header_src, 'r') as myfile:
            data = myfile.read()
            self.headers = data.split(' ')
            self.table = pandas.read_table(data_src, sep = '\t', names = self.headers)

        attrs = ['INCOME1', 'AGE', 'SEX', 'CITIZEN', 'ENGLISH', 'YEARSCH', 'MARITAL', 'FERTIL', 'CLASS', 'INDUSTRY', 'OCCUP', 'HOUR89', 'WEEK89', 'POWSTATE', 'YEARWRK']

        # Only retain attributes that are considered relevant
        self.table = self.table[attrs]

    def getTable(self):
        return self.table

    # Function used to transform the income column into correct labels
    # depending on the given threshold (0: < threshold, 1: >= threshold)
    def produceLabel(self, income, threshold):
        if (income < threshold):
            return 0
        else:
            return 1

    # Function used to map occupation codes into smaller categories
    def classifyOccup(self, code):
        if (0 <= code and code <= 202):
            return 0
        elif (203 <= code and code <= 402):
            return 1
        elif (403 <= code and code <= 472):
            return 2
        elif (503 <= code and code <= 702):
            return 3
        elif (703 <= code and code <= 902):
            return 4
        elif (903 <= code and code <= 908):
            return 5
        else:
            return 6
    
    # Function used to map industry codes into smaller categories
    def classifyIndustry(self, code):
        if (0 <= code and code <= 39):
            return 0
        elif (40 <= code and code <= 59):
            return 1
        elif (60 <= code and code <= 99):
            return 2
        elif (100 <= code and code <= 399):
            return 3
        elif (400 <= code and code <= 499):
            return 4
        elif (500 <= code and code <= 579):
            return 5
        elif (580 <= code and code <= 699):
            return 6
        elif (700 <= code and code <= 720):
            return 7
        elif (721 <= code and code <= 760):
            return 8
        elif (761 <= code and code <= 799):
            return 9
        elif (800 <= code and code <= 811):
            return 10
        elif (812 <= code and code <= 899):
            return 11
        elif (900 <= code and code <= 939):
            return 12
        elif (940 <= code and code <= 991):
            return 13
        else:
            return 14

    def cleanData(self):
        # Producing label based on an inidividual's income, using the threshold of 50000
        self.table['LABEL'] = self.table['INCOME1'].apply(self.produceLabel, args = (50000,))

        # Dropping the income column
        self.table.drop('INCOME1', axis = 1, inplace = True)

        # Categorize occupations into smaller groups
        self.table['OCCUP'] = self.table['OCCUP'].apply(self.classifyOccup)

        # Categorize industry into smaller groups
        self.table['INDUSTRY'] = self.table['INDUSTRY'].apply(self.classifyIndustry)

        # Create dummy variables for categorical variables since svm works better with binary variables
        self.table = pandas.get_dummies(self.table, columns = ['CITIZEN', 'ENGLISH', 'YEARSCH', 'MARITAL', 'FERTIL', 'CLASS', 'INDUSTRY', 'OCCUP', 'POWSTATE', 'YEARWRK']).astype(numpy.int8)

    def evaluate(self, true_label, predicted_label):
        recall = recall_score(true_label, predicted_label)
        precision = precision_score(true_label, predicted_label)
        auc = roc_auc_score(true_label, predicted_label)

        print 'The recall of the model is: ' + str(recall)
        print 'The precision of the model is: ' + str(precision)
        print 'The auc of the model is: ' + str(auc)
        print '\n'

if __name__ == '__main__':
    
    print 'Executing script for predicting salary'

    print 'Reading in data and constructing the matrix'

    mPredictor = predictor('census_data.txt', 'census_header.txt')

    print 'Cleaning and transforming the data'

    mPredictor.cleanData()

    print 'Obtaining different samples for training and testing'
    
    table = mPredictor.getTable()
    training_sample = table.sample(50000)
    testing_sample1 = table.sample(30000)
    testing_sample2 = table.sample(30000)
    testing_sample3 = table.sample(30000)

    # Produce an array of labels, each corresponding to a row in the table containing feature values
    training_labels = numpy.array(training_sample['LABEL'])
    testing_labels1 = numpy.array(testing_sample1['LABEL'])
    testing_labels2 = numpy.array(testing_sample2['LABEL'])
    testing_labels3 = numpy.array(testing_sample3['LABEL'])

    # Use all columns other than the label column for training the classifier
    training_features = training_sample.iloc[:, 0:len(training_sample.columns.values) - 1]
    testing_features1 = testing_sample1.iloc[:, 0:len(testing_sample1.columns.values) - 1]
    testing_features2 = testing_sample2.iloc[:, 0:len(testing_sample2.columns.values) - 1]
    testing_features3 = testing_sample3.iloc[:, 0:len(testing_sample3.columns.values) - 1]

    # Turn the table into 2D numpy arrays so that they can be fed into the SVM classifier
    training_features = numpy.array(training_features)
    testing_features1 = numpy.array(testing_features1)
    testing_features2 = numpy.array(testing_features2)
    testing_features3 = numpy.array(testing_features3)

    print 'Training and testing features and labels are ready. Initializing svm classifier'

    # Initialize the SVM classifier
    classifier = svm.SVC()

    print 'Training in process...'

    # Train the SVM classifier
    classifier.fit(training_features, training_labels)

    print 'Finished training'

    print 'Making predictions using testing samples'

    # Make predictions
    prediction_result1 = classifier.predict(testing_features1)
    prediction_result2 = classifier.predict(testing_features2)
    prediction_result3 = classifier.predict(testing_features3)
    
    # Evaluations

    print 'Evaluating test sample 1'
    mPredictor.evaluate(testing_labels1, prediction_result1)

    print 'Evaluating test sample 2'
    mPredictor.evaluate(testing_labels2, prediction_result2)

    print 'Evaluating test sample 3'
    mPredictor.evaluate(testing_labels3, prediction_result3)
