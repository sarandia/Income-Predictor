import sklearn
import numpy
import pandas
import os
import sklearn.svm
from sklearn import preprocessing
from sklearn import tree
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import SGDClassifier
import time

class predication_engine:
    def load_data(self,sourcefile):
        #get file from HDFS
        print('Getting datasource from Hadoop File System...')
        os.system('hadoop fs -get final_project_data/'+sourcefile+' '+sourcefile)
        #take only attribute columns that are needed
        f=open(sourcefile,'r')
        if sourcefile=='census-income.data':
            datasource='extracted.data'
        else:
            datasource='extracted.test'

        try:
            os.remove(datasource)
        except OSError:
            pass
        print('Extracting useful features...')
        res_file=open(datasource,'a')
        #extract useful features
        for line in f:
            new_line=line.strip().split(",")
            new_line=(new_line[0],new_line[4],new_line[10],new_line[12],new_line[15],new_line[35],new_line[41])
            res_file.write(','.join(new_line)+'\n')
        #load csv into Pandas Dataframe
        df = pandas.read_csv(datasource,names=['age','education','race','gender','employment status','citizenship','salary'])
        array = df.values
        return array

    def train(self,array,classifier):
        print('Learning...')
        #preprocess non-numerical data
        le = preprocessing.LabelEncoder()
        age,education,race,gender,employment,citizenship,salary = self.construct_feature_columns(array)
        le,age,education,race,gender,employment,citizenship,salary = self.encode_input(le,age,education,race,gender,employment,citizenship,salary)
        x=self.construct_input_matrix(age,education,race,gender,employment,citizenship,salary)
        if classifier=='1':
            gnb = tree.DecisionTreeClassifier() 
        elif classifier=='2':
            #gnb = SGDClassifier(loss="log", penalty="l2")
            gnb = sklearn.svm.LinearSVC()
        elif classifier=='3':
            gnb = NearestCentroid()
        gnb.fit(x, salary)
        print('Learning Completed')
        return gnb,le
    
    def construct_feature_columns(self,array):
        age=[]
        education=[]
        race=[]
        gender=[]
        employment=[]
        citizenship=[]
        salary=[]
        #construct feature columns
        for a in array:
            age.append(a[0])
            education.append(a[1])
            race.append(a[2])
            gender.append(a[3])
            employment.append(a[4])
            citizenship.append(a[5])
            salary.append(a[6])
        #there are very few rows (about 5 in 200,000) in which the salary level is unspecified. In this case, it is assumed to be below 50000
        #this is because it could cause the predicator to predict an invalid '3rd type' salary level
        salary_new=[]
        for sal in salary:
            sal=str(sal)
            if sal.find('- 50000') != -1:
                salary_new.append('less than 50000')
            elif sal.find('50000+') != -1:
                salary_new.append('more than 50000')
            else:
                salary_new.append('less than 50000')
        salary=salary_new
        return age,education,race,gender,employment,citizenship,salary
   
    #encode non-numerical input features
    def encode_input(self,le,age,education,race,gender,employment,citizenship,salary):
        education=le.fit_transform(education)
        race=le.fit_transform(race)
        gender=le.fit_transform(gender)
        employment=le.fit_transform(employment)
        citizenship=le.fit_transform(citizenship)
        salary=le.fit_transform(salary)
        return le,age,education,race,gender,employment,citizenship,salary

    def construct_input_matrix(self,age,education,race,gender,employment,citizenship,salary):
        x=[]
        #reconstruct input matrix from encoded features
        for i in xrange(0,len(age)):
            row=(age[i],education[i],race[i],gender[i],employment[i],citizenship[i])
            x.append(row)
        return x
    
    def test(self,array,gnb,le):
        print('Predicting...')
        age,education,race,gender,employment,citizenship,salary=self.construct_feature_columns(array)
        le,age,education,race,gender,employment,citizenship,salary=self.encode_input(le,age,education,race,gender,employment,citizenship,salary)
        x=self.construct_input_matrix(age,education,race,gender,employment,citizenship,salary)
        res=gnb.predict(x)
        #write results to file
        try:
            os.remove('results')
        except OSError:
            pass
        f=open('results','a')
        #compare with expected results
        print('Checking result accuracy...')
        for i in xrange(len(x)):
            f.write(str(res[i])+','+str(salary[i])+'\n')
        return self.compute_metrics(res,salary)
        '''
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        #write results to file
        try:
            os.remove('results')
        except OSError:
            pass
        f=open('results','a')
        #compare with expected results
        print('Checking result accuracy...')
        for i in xrange(len(x)):
            f.write(str(res[i])+','+str(salary[i])+'\n')
            if res[i]==salary[i]:
                if res[i]==0:
                    true_positives+=1
                else:
                    true_negatives+=1
            elif res[i]!=0:
                false_negatives+=1
            else:
                false_positives+=1
        #calculate % of correct predications
        size=len(x)
        true_positive_prob = float(true_positives)/size
        true_negative_prob = float(true_negatives)/size
        false_positive_prob = float(false_positives)/size
        false_negative_prob = float(false_negatives)/size
        precision = true_positive_prob/(true_positive_prob+false_positive_prob)
        recall = true_positive_prob/(true_positive_prob+false_negative_prob)
        F1 = 2*precision*recall/(precision+recall)
        print('---------------------------Result Analysis:---------------------------')
        print('Number of True Positives: '+str(true_positives)+' Percentage of Hits: '+str(100*true_positive_prob)+'%')
        print('Number of True Negatives: '+str(true_negatives)+' Percentage of Hits: '+str(100*true_negative_prob)+'%')
        print('Number of False Positives: '+str(false_positives)+' Percentage of Type I Error: '+str(100*false_positive_prob)+'%')
        print('Number of False Negatives: '+str(false_negatives)+' Percentage of Type II Error: '+str(100*false_negative_prob)+'%')
        print('Precision: '+str(precision))
        print('Recall: '+str(recall))
        print('F1 Score: '+str(F1))
        print('----------------------------------------------------------------------')
        '''
    
    def compute_metrics(self,y_pred,y_true):
        recall = recall_score(y_true,y_pred,average='micro')
        precision = precision_score(y_true,y_pred,average='micro')
        auc = roc_auc_score(y_true,y_pred,average='micro')
        return precision,recall,auc

    def show_results(self,metrics):
        print('Precision: '+str(metrics[0]))
        print('Recall: '+str(metrics[1]))
        print('AUC: '+str(metrics[2]))

    def cleanup(self):
        try:
            os.remove('extracted.data')
            os.remove('extracted.test')
            os.remove('census-income.data')
            os.remove('census-income.test')
        except OSError:
            pass
    
    def __init__(self,source_data,test_data):
         self.sourcedata=source_data
         self.testdata=test_data

    def run(self):
        print("Choose one of the following classifiers:\n1.Decision Tree\n2.SVM\n3.Nearest Centroid\nType in the number to select")
        user_input = raw_input()
        if user_input != '1' and user_input != '2' and user_input !='3':
            print('Invalid classifier selection')
            sys.exit(1)
        start_time=time.time() 
        training_set = self.load_data(self.sourcedata)
        learning_start=time.time()
        classifier,label_encoder = self.train(training_set,user_input)
        learning_time=time.time()-learning_start
        print('Learning took '+str(learning_time)+' seconds')
        test_set = self.load_data(self.testdata)
        metrics=self.test(test_set,classifier,label_encoder)
        self.show_results(metrics)
        self.cleanup()
        print('Program finished. The run took '+str(time.time()-start_time)+' seconds')

if __name__== '__main__':
   p = predication_engine('census-income.data','census-income.test')
   p.run() 
