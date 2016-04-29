import numpy
import pandas
import csv

'''
with open('ss13pusb.csv', 'rb') as csvfile:
    dataSource = csv.reader(csvfile)
    cnt = 0
    while cnt < 10:
        print dataSource.next()
        cnt += 1
'''

# get the headers (in the first line of the data)
# headers = []
# with open('ss13pubs.csv', 'rb') as csvfile:
#    data_source = csv.reader(csvfile)
#    headers = data_source.next()

# the list of features to use
# [Record Type (RT), Housing serial number (SERIALNO), State (ST), Age (AGEP), 
# Citizenship Status (CIT), Class of worker (COW), Mean of transportation to work (JWTR),
# Marital status (MAR), Educational attainment (SCHL), Sex (SEX) Wages or Salary income 
# past 12 months (WAGP), Usual hours worked per week past 12 months (WKHP), WKW (weeks 
# work during last year), Recoded field of degree (FOD1P)]
features = ['RT', 'SERIALNO', 'ST', 'AGEP', 'CIT', 'JWTR', 'MAR', 'SCHL', 'SEX', \
            'WAGP', 'WKHP', 'WKW', 'FOD1P']

# Read data from file
dataframe = pandas.read_csv('ss13pusb.csv', na_values='', header=0, \
                              nrows=10)

data_array = dataframe.values

'''
# test parsing in input
data_list = data_array.tolist()

for person in data_list:
    print person
'''
