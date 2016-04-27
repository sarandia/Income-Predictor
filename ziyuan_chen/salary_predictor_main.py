import os,sys

print("How many features would you like to use? Enter 8 or 10")
a = raw_input()

while True:
    if a=='8':
        os.system('python predict_salary.py')
        sys.exit(0)
    elif a=='10':
        os.system('python predict_salary_10_features.py')
        sys.exit(0)
    else:
        print('Invalid option. Please enter again (8 or 10):')
        a = raw_input()
