import numpy
import pandas


class predictor:

    def __init__(self):
        self.table = []

    def load_data(self, filename):
        
        # the list of features of interest to use
        # [Record Type (RT), Housing serial number (SERIALNO), State (ST), Age (AGEP), 
        # Citizenship Status (CIT), Class of worker (COW), Mean of transportation to work (JWTR),
        # Marital status (MAR), Educational attainment (SCHL), Sex (SEX), Wages or Salary income 
        # past 12 months (WAGP), Usual hours worked per week past 12 months (WKHP), WKW (weeks 
        # work during last year), Recoded field of degree (FOD1P)]
        features = ['RT', 'SERIALNO', 'ST', 'AGEP', 'CIT', 'JWTR', 'MAR', 'SCHL', 'SEX', \
                    'WAGP', 'WKHP', 'WKW', 'FOD1P']

        # Read data from file
        dataframe = pandas.read_csv(filename, na_values='', header=0, \
                                    usecols=features, nrows=100)

        self.table = dataframe.values

    def test_data(self):
        data_list = self.table.tolist()

        for person in data_list:
            print person

    def get_table(self):
        return self.table

if __name__ == '__main__':
    p = predictor()
    p.load_data('ss13pus.csv')
    p.test_data()
