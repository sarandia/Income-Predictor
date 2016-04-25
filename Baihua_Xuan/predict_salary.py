import numpy
import scipy
import sklearn
import pandas
    
class predict_salary:
    def __init__(self, data_src, header_src):
        self.header = []

        with open(header_src, 'r') as myfile:
            data = myfile.read().replace('\n', '')
            self.header = data.split(', ')
            print self.header

        self.table = pandas.read_csv(data_src, self.header)


if __name__ == '__main__':
    print 'Executing script for predicting salary'
    predictor = predict_salary()
