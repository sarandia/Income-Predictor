import numpy
import scipy
import sklearn
import pandas
from sklearn import svm
from sklearn import tree
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from z_predictor import z_predictor
from h_predictor import h_predictor
from b_predictor import b_predictor
def evaluate(true_label,predicted_label):
	recall=recall_score(true_label,predicted_label)
	precision = precision_score(true_label,predicted_label)
	auc = roc_auc_score(true_label,predicted_label)
	print'The recall of the model is:'+str(recall)
	print 'The precision of the model is:' +str(precision)
	print 'The auc of the model is:' +str(auc)
def run():
	p1=b_predictor('census_data.txt','census_header.txt')
	p2=z_predictor('1','2')
	p3=h_predictor()
	p1.cleanData()
	x1,y1=p1.getData()
	x2,y2=p2.combined_predict()
	x3,y3=p3.format_data_combine('ss13pusb.csv')
	feature1=numpy.asarray(x1)
	feature2=numpy.asarray(x2)
	feature3=numpy.asarray(x3)
	#test label
	l_test1=y1[range(0,5000),]
	l_test2=numpy.asarray(y2)[range(0,5000),]
	l_test=numpy.concatenate((l_test1,l_test2))
	
	#test feature
	f_test1=x1[range(0,5000),]
	f_test2=numpy.asarray(x2)[range(0,5000),]
	f_test=numpy.concatenate((f_test1,f_test2))
	#trainning feature
	 #trainning label
        l_train1=y1[range(5001,len(y1)),]
        l_train2=numpy.asarray(y2)[range(5001,len(y1)),]
        l_train=numpy.concatenate((l_train1,l_train2))

        #trainning feature
        f_train1=x1[range(5001,len(x1)),]
        f_train2=numpy.asarray(x2)[range(5001,len(x1)),]
        f_train=numpy.concatenate((f_train1,f_train2))
	
	print 'Training in process... (DT)'
	dec_tree = tree.DecisionTreeClassifier()
	dec_tree.fit(f_train, l_train)
	tree_res1 = dec_tree.predict(f_test)
	print 'Evaluating test sample 1 (svm)'
        l_test_new=[]
        tree_res1_new=[]
        for i in range(len(tree_res1)):
            if tree_res1[i]==1:
                tree_res1_new.append(0)
            else:
                tree_res1_new.append(1)
            if l_test[i]==1:
                l_test_new.append(0)
            else:
                l_test_new.append(1)
        evaluate(l_test_new, tree_res1_new)
	print 'Training in process... (svm)'
	classifier1 = svm.SVC()
	classifier1.fit(f_train, l_train)
	svm_res1 = classifier1.predict(f_test)
	print 'Evaluating test sample 1 (svm)'
	evaluate(l_test, svm_res1)
if __name__== '__main__':
	run()
