import sklearn
import numpy
import pandas 
import os
import sklearn.svm
from sklearn import svm
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


class predictor:
	def load_data(self,source):
			df=pandas.read_csv('T1.csv',usecols=[1,4,5,6,7,8,11,10],names=['Position','Location','Time','Permanent','Company','Type','Salary','Source'])
			df = df[['Position','Location','Time','Permanent','Company','Type','Source','Salary']]
			df.drop(df.index[[0]],inplace=True)			
			return df
	

	def train(self,table):
		le=preprocessing.LabelEncoder()
		table['Label']=table.apply(f,axis=1)
		table.drop('Salary',axis=1,inplace=True)
		mtable=encode_input(table,le)
		return mtable

	def __init__(self,source_data):
		self.sourcedata=source_data

	def run(self):
		trainning=self.load_data(self.sourcedata)
		trainning=trainning.iloc[numpy.random.permutation(len(trainning))]
		start_time=time.time() 
		mtable=train(trainning)
		tr=mtable.iloc[0:100000]
		tr2=mtable.iloc[0:10000]
		tr_feature=tr.iloc[:,0:len(tr.columns.values)-1]
		tr2_feature=tr2.iloc[:,0:len(tr2.columns.values)-1]
		tr_label=numpy.array(tr['Label'])
		tr2_label=numpy.array(tr2['Label'])
		

		ts_sam1 = mtable.iloc[100001:120000]
		ts_sam1_label= numpy.array(ts_sam1['Label'])
		ts_sam1_feature=ts_sam1.iloc[:,0:len(ts_sam1.columns.values)-1]
		ts_sam2=mtable.iloc[50000:52000]
		ts_sam2_label=numpy.array(ts_sam2['Label'])
		ts_sam2_feature=ts_sam2.iloc[:,0:len(ts_sam2.columns.values)-1]
		# Turn the table into 2D numpy arrays so that they can be fed into the SVM classifier
		tr_feature=numpy.array(tr_feature)
		print 'Training and testing features and labels are ready. Initializing tree classifier'
		classifier = tree.DecisionTreeClassifier()
		print 'Training in process... (Tree)'
    		# Train the SVM classifier
    		classifier.fit(tr_feature, tr_label)
		print 'Finished training (Tree)'
		svm_res1=classifier.predict(ts_sam1_feature)		
		print 'Evaluating test sample 1 (Tree)'
		evaluate(ts_sam1_label, svm_res1)
		print 'Training in process... (SVM)'
		classifier = svm.SVC()

		classifier.fit(tr2_feature,tr2_label)

		svm_res2=classifier.predict(ts_sam2_feature)
		print 'Evaluating test sample 1 (Tree)'
		evaluate(ts_sam2_label,svm_res2)

		
def shuffle(df, n=1, axis=0):     
        df = df.copy()
  	for _ in range(n):
        	df.apply(numpy.random.shuffle, axis=axis)
        return df
def train(table):
                le=preprocessing.LabelEncoder()
                table['Label']=table.apply(f,axis=1)
                table.drop('Salary',axis=1,inplace=True)
                mtable=encode_input(table,le)
                mtable.to_csv('tem.csv', sep='\t')
                return mtable			
def evaluate(true_label,predicted_label):
	recall=recall_score(true_label,predicted_label)
	precision = precision_score(true_label,predicted_label)
	auc = roc_auc_score(true_label,predicted_label)
	print'The recall of the model is:'+str(recall)
	print 'The precision of the model is:' +str(precision)
	print 'The auc of the model is:' +str(auc)

def encode_input(table,le):
	table['Position']=le.fit_transform(table['Position'])
	table['Location']=le.fit_transform(table['Location'])
	table['Time']=le.fit_transform(table['Time'])
	table['Permanent']=le.fit_transform(table['Permanent'])
	table['Company']=le.fit_transform(table['Company'])
	table['Type']=le.fit_transform(table['Type'])
	table['Source']=le.fit_transform(table['Source'])
	return table
def f(row):
	if (int(row['Salary'])<30000):
		return 0
	else:
		return 1

if __name__== '__main__':
   p = predictor('T1.csv')
   p.run() 
		

