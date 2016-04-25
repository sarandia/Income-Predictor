import numpy
import scipy
import sklearn
import pandas
    
class predict_salary:
    def __init__(self, data_src, header_src):
        
        if (header_src is None):
            # First line is the header
            self.table = pandas.read_csv(data_src)
        else:
            with open(header_src, 'r') as myfile:
                data = myfile.read().replace('\n', '')
                self.headers = data.split(', ')
                print self.headers
                self.table = pandas.read_csv(data_src, self.header)

        #print self.table

    

if __name__ == '__main__':
    print 'Executing script for predicting salary'
    predictor = predict_salary('trial.txt', None)
