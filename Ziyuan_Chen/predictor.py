import sklearn
import numpy
import pandas
import os
import sklearn.svm
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn import neighbors
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
            new_line=(new_line[0],new_line[2],new_line[4],new_line[5],new_line[10],new_line[12],new_line[15],new_line[39],new_line[41])
            res_file.write(','.join(new_line)+'\n')
        #load csv into Pandas Dataframe
        df = pandas.read_csv(datasource,names=['age','industry code','occupation','education','race','gender','employment status','weeks worked in a year','salary'])
        array = df.values
        return array

    def train(self,array,classifier):
        print('Learning...')
        #preprocess non-numerical data
        le = preprocessing.LabelEncoder()
        age,industry,occupation,education,race,gender,employment,weeks_worked,salary = self.construct_feature_columns(array)
        le,age,industry,occupation,education,race,gender,employment,weeks_worked,salary = self.encode_input(le,age,industry,occupation,education,race,gender,employment,weeks_worked,salary)
        x=self.construct_input_matrix(age,industry,occupation,education,race,gender,employment,weeks_worked,salary)
        if classifier=='1':
            gnb = tree.DecisionTreeClassifier() 
        elif classifier=='2':
            #gnb = SGDClassifier(loss="log", penalty="l2")
            gnb = sklearn.svm.LinearSVC()
        elif classifier=='3':
            gnb = neighbors.KNeighborsClassifier(15,weights='distance')
        gnb.fit(x, salary)
        print('Learning Completed')
        return gnb,le
    
    def construct_feature_columns(self,array):
        age=[]
        industry=[]
        occupation=[]
        education=[]
        race=[]
        gender=[]
        employment=[]
        weeks_worked=[]
        salary=[]
        #construct feature columns
        for a in array:
            age.append(a[0])
            industry.append(a[1])
            occupation.append(a[2])
            education.append(a[3])
            race.append(a[4])
            gender.append(a[5])
            employment.append(a[6])
            weeks_worked.append(a[7])
            salary.append(a[8])
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
        return age,industry,occupation,education,race,gender,employment,weeks_worked,salary
   
    #encode non-numerical input features
    def encode_input(self,le,age,industry,occupation,education,race,gender,employment,weeks_worked,salary):
        education=le.fit_transform(education)
        industry=le.fit_transform(industry)
        occupation=le.fit_transform(occupation)
        race=le.fit_transform(race)
        gender=le.fit_transform(gender)
        employment=le.fit_transform(employment)
        weeks_worked=le.fit_transform(weeks_worked)
        salary=le.fit_transform(salary)
        return le,age,industry,occupation,education,race,gender,employment,weeks_worked,salary

    def construct_input_matrix(self,age,industry,occupation,education,race,gender,employment,weeks_worked,salary):
        x=[]
        #reconstruct input matrix from encoded features
        for i in xrange(0,len(age)):
            row=(age[i],industry[i],occupation[i],education[i],race[i],gender[i],employment[i],weeks_worked[i])
            x.append(row)
        return x
    
    def test(self,array,gnb,le):
        print('Predicting...')
        age,industry,occupation,education,race,gender,employment,weeks_worked,salary=self.construct_feature_columns(array)
        le,age,industry,occupation,education,race,gender,employment,weeks_worked,salary=self.encode_input(le,age,industry,occupation,education,race,gender,employment,weeks_worked,salary)
        x=self.construct_input_matrix(age,industry,occupation,education,race,gender,employment,weeks_worked,salary)
        res=gnb.predict(x)
        #compare with expected results
        print('Checking result accuracy...')
        return self.compute_metrics(res,salary)
    
    def compute_metrics(self,y_pred,y_true):
        #define 0 as the positive case (salary below 50000)
        temp1=[]
        temp2=[]
        for x in y_pred:
            if x==1:
                temp1.append(0)
            else:
                temp1.append(1)
        for y in y_true:
            if y==1:
                temp2.append(0)
            else:
                temp2.append(1)
        y_pred=temp1
        y_true=temp2
        accuracy = accuracy_score(y_true,y_pred)
        recall = recall_score(y_true,y_pred,average='micro')
        precision = precision_score(y_true,y_pred,average='micro')
        auc = roc_auc_score(y_true,y_pred,average='micro')
        return accuracy,precision,recall,auc

    def show_results(self,metrics):
        print('Accuracy: '+str(metrics[0]))
        print('Precision: '+str(metrics[1]))
        print('Recall: '+str(metrics[2]))
        print('AUC: '+str(metrics[3]))

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
        print("Choose one of the following classifiers:\n1.Decision Tree\n2.SVM\n3.kNN\nType in the number to select")
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
