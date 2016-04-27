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
            new_line=(new_line[0],new_line[2],new_line[4],new_line[7],new_line[10],new_line[12],new_line[15],new_line[34],new_line[35],new_line[39],new_line[41])
            res_file.write(','.join(new_line)+'\n')
        #load csv into Pandas Dataframe
        df = pandas.read_csv(datasource,names=['age','industry code','education','marital status','race','gender','employment status','country of birth','citizenship','weeks worked in a year','salary'])
        array = df.values
        return array

    def train(self,array,classifier):
        print('Learning...')
        #preprocess non-numerical data
        le = preprocessing.LabelEncoder()
        age,industry,education,marital,race,gender,employment,country_birth,citizenship,weeks_worked,salary = self.construct_feature_columns(array)
        le,age,industry,education,marital,race,gender,employment,country_birth,citizenship,weeks_worked,salary = self.encode_input(le,age,industry,education,marital,race,gender,employment,country_birth,citizenship,weeks_worked,salary)
        x=self.construct_input_matrix(age,industry,education,marital,race,gender,employment,country_birth,citizenship,weeks_worked,salary)
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
        industry=[]
        education=[]
        marital=[]
        race=[]
        gender=[]
        employment=[]
        country_birth=[]
        citizenship=[]
        weeks_worked=[]
        salary=[]
        #construct feature columns
        for a in array:
            age.append(a[0])
            industry.append(a[1])
            education.append(a[2])
            marital.append(a[3])
            race.append(a[4])
            gender.append(a[5])
            employment.append(a[6])
            country_birth.append(a[7])
            citizenship.append(a[8])
            weeks_worked.append(a[9])
            salary.append(a[10])
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
        return age,industry,education,marital,race,gender,employment,country_birth,citizenship,weeks_worked,salary
   
    #encode non-numerical input features
    def encode_input(self,le,age,industry,education,marital,race,gender,employment,country_birth,citizenship,weeks_worked,salary):
        education=le.fit_transform(education)
        industry=le.fit_transform(industry)
        marital=le.fit_transform(marital)
        race=le.fit_transform(race)
        gender=le.fit_transform(gender)
        employment=le.fit_transform(employment)
        country_birth=le.fit_transform(country_birth)
        citizenship=le.fit_transform(citizenship)
        weeks_worked=le.fit_transform(weeks_worked)
        salary=le.fit_transform(salary)
        return le,age,industry,education,marital,race,gender,employment,country_birth,citizenship,weeks_worked,salary

    def construct_input_matrix(self,age,industry,education,marital,race,gender,employment,country_birth,citizenship,weeks_worked,salary):
        x=[]
        #reconstruct input matrix from encoded features
        for i in xrange(0,len(age)):
            row=(age[i],industry[i],education[i],marital[i],race[i],gender[i],employment[i],country_birth[i],citizenship[i],weeks_worked[i])
            x.append(row)
        return x
    
    def test(self,array,gnb,le):
        print('Predicting...')
        age,industry,education,marital,race,gender,employment,country_birth,citizenship,weeks_worked,salary=self.construct_feature_columns(array)
        le,age,industry,education,marital,race,gender,employment,country_birth,citizenship,weeks_worked,salary=self.encode_input(le,age,industry,education,marital,race,gender,employment,country_birth,citizenship,weeks_worked,salary)
        x=self.construct_input_matrix(age,industry,education,marital,race,gender,employment,country_birth,citizenship,weeks_worked,salary)
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
            #os.remove('census-income.data')
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
