import numpy
import scipy
import sklearn
import pandas
from sklearn import svm
from sklearn import tree
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
    
# Class predictor
# Used to analyze and perform machine learning tasks on the 1990 census data
# obtained from the UCI Machine Learning Repository
 
class b_predictor:
    
    # Constructor
    def __init__(self, data_src, header_src):
        
        # Read in the parsed header file and construct a pandas table
        with open(header_src, 'r') as myfile:
            data = myfile.read()
            self.headers = data.split(' ')
            self.table = pandas.read_table(data_src, sep = '\t', names = self.headers)

        attrs = ['INCOME1', 'AGE', 'SEX', 'CITIZEN', 'ENGLISH', 'YEARSCH', 'MARITAL', 'FERTIL', 'CLASS', 'INDUSTRY', 'OCCUP', 'HOUR89', 'WEEK89', 'POWSTATE', 'YEARWRK']

        # Only retain attributes that are considered relevant
        self.table = self.table[attrs]

    # Getter: table field
    def getTable(self):
        return self.table

    # Getter: get original table withou dummy variables
    def getWhole(self):
        return self.whole

    # Function used to transform the income column into correct labels
    # depending on the given threshold (0: < threshold, 1: >= threshold)
    def produceLabel(self, income, threshold):
        if (income < threshold):
            return 0
        else:
            return 1

    # Function used to map occupation codes into smaller categories
    # Mappings between occupation and code can be found at:
    # http://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990raw.coding.htm
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
    # Mappings between industry and code can be found at:
    # http://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990raw.coding.htm
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

    # Function to shrink weeks worked last year into smaller categories
    # Used when combining data
    def classifyWeek(self, code):
        if code < 14:
            return 6
        elif (14 <= code and code <= 26):
            return 5
        elif (27 <= code and code <= 39):
            return 4
        elif (40 <= code and code <= 47):
            return 3
        elif (48 <= code and code <= 49):
            return 2
        else:
            return 1

    # Function for cleaning the data set
    # Producing labels based on one's income and append it to table
    # Dropping the income column
    # Transform occupation and industry code into smaller categories using helper functions
    def cleanData(self):
        # Producing label based on an inidividual's income, using the threshold of 50000
        self.table['LABEL'] = self.table['INCOME1'].apply(self.produceLabel, args = (50000,))

        # Dropping the income column
        self.table.drop('INCOME1', axis = 1, inplace = True)

        # Categorize occupations into smaller groups
        self.table['OCCUP'] = self.table['OCCUP'].apply(self.classifyOccup)

        # Categorize industry into smaller groups
        self.table['INDUSTRY'] = self.table['INDUSTRY'].apply(self.classifyIndustry)

        # Save the whole table without dummy variables
        self.whole = self.table

        # Create dummy variables for categorical variables since svm works better with binary variables
        self.table = pandas.get_dummies(self.table, columns = ['CITIZEN', 'ENGLISH', 'YEARSCH', 'MARITAL', 'FERTIL', 'CLASS', 'INDUSTRY', 'OCCUP', 'POWSTATE', 'YEARWRK']).astype(numpy.int8)

    # Function for evaluating a model
    # Metrics used: recall, precision and area under the roc curve
    def evaluate(self, true_label, predicted_label):
        recall = recall_score(true_label, predicted_label)
        precision = precision_score(true_label, predicted_label)
        auc = roc_auc_score(true_label, predicted_label)

        print 'The recall of the model is: ' + str(recall)
        print 'The precision of the model is: ' + str(precision)
        print 'The auc of the model is: ' + str(auc)
        print '\n'

    # Function to return portion of the table used for aggregated data
    def getData(self):
        data = self.whole.iloc[0:50000]
        features = data[['AGE', 'SEX', 'YEARSCH', 'INDUSTRY', 'WEEK89']]
        features['WEEK89'] = features['WEEK89'].apply(self.classifyWeek)
        labels = data['LABEL']
        
        print features, labels

        return numpy.array(features), numpy.array(labels)

if __name__ == '__main__':
    
    print 'Executing script for predicting salary'

    print 'Reading in data and constructing the matrix'

    mPredictor = predictor('census_data.txt', 'census_header.txt')

    print 'Cleaning and transforming the data'

    mPredictor.cleanData()

    print 'Obtaining different samples for training and testing'
    
    # Use getWhole() for dec_tree, getTable() for svm
    table = mPredictor.getTable()
    whole_table = mPredictor.getWhole().iloc[0:2000000]

    # Training and testing samples for svm
    tr_sam = table.iloc[0:1000000].sample(50000)
    ts_sam1 = table.iloc[1000000:2000000].sample(30000)
    ts_sam2 = table.iloc[1000000:2000000].sample(30000)
    ts_sam3 = table.iloc[1000000:2000000].sample(30000)

    # Testing samples for dec_tree
    ts_sam1_w = mPredictor.getWhole().iloc[2000000:].sample(300000)
    ts_sam2_w = mPredictor.getWhole().iloc[2000000:].sample(300000)
    ts_sam3_w = mPredictor.getWhole().iloc[2000000:].sample(300000)

    # Produce arrays of labels, each corresponding to the label of a row in the table
    tr_lab = numpy.array(tr_sam['LABEL'])
    ts_lab1 = numpy.array(ts_sam1['LABEL'])
    ts_lab2 = numpy.array(ts_sam2['LABEL'])
    ts_lab3 = numpy.array(ts_sam3['LABEL'])
    
    # Labels for dec_tree
    tr_lab_w = numpy.array(whole_table['LABEL'])
    ts_lab1_w = numpy.array(ts_sam1_w['LABEL'])
    ts_lab2_w = numpy.array(ts_sam2_w['LABEL'])
    ts_lab3_w = numpy.array(ts_sam3_w['LABEL'])


    # Extract all columns other than the label column for training the classifier
    tr_f = tr_sam.iloc[:, 0:len(tr_sam.columns.values) - 1]
    ts_f1 = ts_sam1.iloc[:, 0:len(ts_sam1.columns.values) - 1]
    ts_f2 = ts_sam2.iloc[:, 0:len(ts_sam2.columns.values) - 1]
    ts_f3 = ts_sam3.iloc[:, 0:len(ts_sam3.columns.values) - 1]

    tr_f_w = whole_table.iloc[:, 0:len(whole_table.columns.values) - 1]
    ts_f1_w = ts_sam1_w.iloc[:, 0:len(ts_sam1_w.columns.values) - 1]
    ts_f2_w = ts_sam2_w.iloc[:, 0:len(ts_sam2_w.columns.values) - 1]
    ts_f3_w = ts_sam3_w.iloc[:, 0:len(ts_sam3_w.columns.values) - 1]


    # Turn the table into 2D numpy arrays so that they can be fed into the SVM classifier
    tr_f = numpy.array(tr_f)
    ts_f1 = numpy.array(ts_f1)
    ts_f2 = numpy.array(ts_f2)
    ts_f3 = numpy.array(ts_f3)
    
    tr_f_w = numpy.array(tr_f_w)
    ts_f1_w = numpy.array(ts_f1_w)
    ts_f2_w = numpy.array(ts_f2_w)
    ts_f3_w = numpy.array(ts_f3_w)

    print 'Training and testing features and labels are ready. Initializing svm classifier'

    # Initialize the SVM classifier
    classifier = svm.SVC()

    print 'Training in process... (svm)'

    # Train the SVM classifier
    classifier.fit(tr_f, tr_lab)

    print 'Finished training (svm)'

    print 'Making predictions using testing samples (svm)'

    # Make predictions
    svm_res1 = classifier.predict(ts_f1)
    svm_res2 = classifier.predict(ts_f2)
    svm_res3 = classifier.predict(ts_f3)
    
    # Evaluations

    print 'Evaluating test sample 1 (svm)'
    mPredictor.evaluate(ts_lab1, svm_res1)

    print 'Evaluating test sample 2 (svm)'
    mPredictor.evaluate(ts_lab2, svm_res2)

    print 'Evaluating test sample 3 (svm)'
    mPredictor.evaluate(ts_lab3, svm_res3)


    print '---------------------------------------------------\n'

    print 'Now switching to using decision tree as the classfier'

    print 'Initialzing decision tree classifier' 

    dec_tree = tree.DecisionTreeClassifier()

    print 'Training in process... (decision tree)'

    print 'Finished training (decision tree)'

    print 'Making predictions using the whole table (decision tree)'

    # Training the decision tree classifier
    # This is trained using the whole table

    dec_tree.fit(tr_f_w, tr_lab_w)

    tree_res1 = dec_tree.predict(ts_f1_w)
    tree_res2 = dec_tree.predict(ts_f2_w)
    tree_res3 = dec_tree.predict(ts_f3_w)

    # Evaluations

    print 'Evaluating prediction over the entire dataset (decision tree) \n'

    print 'Evaluating sample1 (dec_tree)'
    mPredictor.evaluate(ts_lab1_w, tree_res1)

    print 'Evaluating sample2 (dec_tree)'
    mPredictor.evaluate(ts_lab2_w, tree_res2)

    print 'Evaluating sample3 (dec_tree)'
    mPredictor.evaluate(ts_lab3_w, tree_res3)

    
