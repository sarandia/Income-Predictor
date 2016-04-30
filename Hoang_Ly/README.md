Predictor of salary based on other demographics data

The source is the 2013 American Community Survey (ACS), US Cencus Bureau
PUMS DATA

Link: http://www2.census.gov/acs2013_1yr/pums/csv_pus.zip
The file size is 1.4Gb once unzipped.

The detailed exaplation of the headers are given in the file at http://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict13.pdf

The features of interest in the prediction are:
- Record Type (RT)
- Housing serial number (SERIALNO)
- State (ST)
- Age (AGEP) 
- Citizenship Status (CIT)
- Mean of transportation to work (JWTR)
- Marital status (MAR)
- Educational attainment (SCHL)
- Sex (SEX) 
- Wages or Salary income past 12 months (WAGP)
- Usual hours worked per week past 12 months (WKHP)
- WKW (weeks work during past 12 months 
- Recoded field of degree (FOD1P)

Program description:
- Loads data to matrix using pandas

- Train and predict on each single feature with two type of technique:
  + Classification (Naive Bayes and Decision Tree): for this we also need have
  salary array (target array) divided into discrete levels
  + Linear Regression: for this we have salary array as regular array of float values 
  
  For each prediction, format the matrix to get rid of null or NaN values

- Train and predict on two most accurate single features: WKHP and WKW 