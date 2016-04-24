import sklearn
import numpy
import pandas
import os
import sklearn.svm
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import time

class predict_salary:
    def load_data(sourcefile):
        
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

    def train(array):
        print('Learning...')
        #preprocess non-numerical data
        le = preprocessing.LabelEncoder()
        age=[]
        education=[]
        race=[]
        gender=[]
        employment=[]
        citizenship=[]
        salary=[]
        #construct arrays of values of each column
        for a in array:
            age.append(a[0])
            education.append(a[1])
            race.append(a[2])
            gender.append(a[3])
            employment.append(a[4])
            citizenship.append(a[5])
            salary.append(a[6])
        #preprocess values to encode string features
        education=le.fit_transform(education)
        race=le.fit_transform(race)
        gender=le.fit_transform(gender)
        employment=le.fit_transform(employment)
        citizenship=le.fit_transform(citizenship)
        salary=le.fit_transform(salary)
        x=[]
        #put arrays back into rows
        for i in xrange(0,len(age)):
            row=(age[i],education[i],race[i],gender[i],employment[i],citizenship[i])
            x.append(row)
        #gnb = sklearn.svm.LinearSVC()
        gnb = tree.DecisionTreeClassifier() 
        gnb.fit(x, salary)
        print('Learning Completed')
        return gnb,le

    def test(array,gnb,le):
        print('Predicting...')
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
        #encode non-numeric features 
        education=le.fit_transform(education)
        race=le.fit_transform(race)
        gender=le.fit_transform(gender)
        employment=le.fit_transform(employment)
        citizenship=le.fit_transform(citizenship)
        salary=le.fit_transform(salary)
        x=[]
        #reconstruct input matrix from encoded features
        for i in xrange(0,len(age)):
            row=(age[i],education[i],race[i],gender[i],employment[i],citizenship[i])
            x.append(row)
        res=gnb.predict(x)
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
        return 
    
    def cleanup():
        try:
            os.remove('extracted.data')
            os.remove('extracted.test')
            #os.remove('census-income.data')
            os.remove('census-income.test')
        except OSError:
            pass

    if __name__ == "__main__":
        start_time=time.time() 
        training_set = load_data('census-income.data')
        learning_start=time.time()
        classifier,label_encoder = train(training_set)
        learning_time=time.time()-learning_start
        print('Learning took '+str(learning_time)+' seconds')
        test_set = load_data('census-income.test')
        test(test_set,classifier,label_encoder)
        cleanup()
        print('Program finished. The run took '+str(time.time()-start_time)+' seconds')
