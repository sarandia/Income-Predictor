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
from sklearn.metrics import accuracy_score
from sklearn import neighbors
import time

class z_predictor:
    def combined_predict(self):#provide input matrix for combined prediction
        #take only attribute columns that are needed
        f=open('census-income.data','r')
        res_file=open('temp_data','a')
        #extract useful features
        for line in f:
            new_line=line.strip().split(",")
            new_line=(new_line[0],new_line[12],new_line[4],new_line[2],new_line[39],new_line[41])
            res_file.write(','.join(new_line)+'\n')
        #load csv into Pandas Dataframe
        df = pandas.read_csv('temp_data',names=['age','sex','education','industry code','weeks worked in a year','salary'])
        array = df.values[0:50000]
        age,sex,education,industry,weeks_worked,salary = self.construct_feature_columns_combined_predict(array)
        weeks_worked_new=[]
        #categorize weeks worked for combined predication
        for code in weeks_worked:
            weeks_worked_new.append(self.classifyWeek(code))
        weeks_worked = weeks_worked_new
        #encode education levels
        education_new=[]
        for edu in education:
            education_new.append(self.encode_education(edu))
        education=education_new
        #encoded sex. Male=0, Female=1
        sex_new=[]
        for s in sex:
            sex_new.append(self.encode_sex(s))
        sex=sex_new
        os.remove('temp_data')
        #construct input matrix
        x=[]
        for i in xrange(0,len(age)):
            row=(age[i],sex[i],education[i],industry[i],weeks_worked[i])
            x.append(row)
        y=salary
        '''
        c=open('see_result','a')
        for i in range(len(x)):
            c.write(str(x[i])+"..."+str(y[i])+"\n")
        '''
        return x,y

    def encode_sex(self,sex):
        if sex.find('Male')!=-1:
            return 0
        else:
            return 1

    def encode_education(self,edu): #for combined predication
        if edu.find('Children')!=-1:
            return 0
        elif edu.find('7th and 8th grade')!=-1:
            return 5
        elif edu.find('9th grade')!=-1:
            return 6
        elif edu.find('10th grade')!=-1:
            return 7
        elif edu.find('High school graduate')!=-1:
            return 10
        elif edu.find('11th grade')!=-1:
            return 8
        elif edu.find('12th grade no diploma')!=-1:
            return 9
        elif edu.find('5th or 6th grade')!=-1:
            return 5
        elif edu.find('Less than 1st grade')!=-1:
            return 3
        elif edu.find('Bachelors')!=-1:
            return 14
        elif edu.find('1st 2nd 3rd or 4th grade')!=-1:
            return 4
        elif edu.find('Some college but no degree')!=-1:
            return 11
        elif edu.find('Masters degree')!=-1:
            return 15
        elif edu.find('Associates degree-occup')!=-1:
            return 12
        elif edu.find('Associates degree-academic')!=-1:
            return 13
        elif edu.find('Doctorate')!=-1:
            return 17
        elif edu.find('Prof school degree')!=-1:
            return 16
        else:
            return 1

    def classifyWeek(self, code): #written by Baihua Xuan, for combined predication
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
    
    def construct_feature_columns_combined_predict(self,array):
        age=[]
        sex=[]
        education=[]
        industry=[]
        weeks_worked=[]
        salary=[]
        #construct feature columns
        for a in array:
            age.append(a[0])
            sex.append(a[1])
            education.append(a[2])
            industry.append(a[3])
            weeks_worked.append(a[4])
            salary.append(a[5])
        #there are very few rows (about 5 in 200,000) in which the salary level is unspecified. In this case, it is assumed to be below 50000
        #this is because it could cause the predicator to predict an invalid '3rd type' salary level
        salary_new=[]
        for sal in salary:
            sal=str(sal)
            if sal.find('- 50000') != -1:
                salary_new.append(0)
            elif sal.find('50000+') != -1:
                salary_new.append(1)
            else:
                salary_new.append(0)
        salary=salary_new
        return age,sex,education,industry,weeks_worked,salary    

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
