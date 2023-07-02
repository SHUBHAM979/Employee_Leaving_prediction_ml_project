# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# dataset = pd.read_csv('hiring.csv')

# dataset['experience'].fillna(0, inplace=True)

# dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

# X = dataset.iloc[:, :3]

# #Converting words to integer values
# def convert_to_int(word):
#     word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
#                 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
#     return word_dict[word]

# X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

# y = dataset.iloc[:, -1]

# #Splitting Training and Test Set
# #Since we have a very small dataset, we will train our model with all availabe data.

# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()

# #Fitting model with trainig data
# regressor.fit(X, y)



employee_data = pd.read_csv('HR.csv')

employee_data.rename(columns = {'sales' : 'department'},inplace=True)

employee_data.department = np.where(employee_data.department == 'support' , 'technical', employee_data.department)
employee_data.department = np.where(employee_data.department == 'IT' , 'technical', employee_data.department)


cat_vars=['department','salary']
for var in cat_vars:
#     cat_list='var'+'_'+var
    cat_list = pd.get_dummies(employee_data[var], prefix=var)
    hr1=employee_data.join(cat_list)
    employee_data=hr1

employee_data.drop(employee_data.columns[[8,9]], axis =1, inplace =True)

employee_var= employee_data.columns.values.tolist()

y = ['left']
x = [i for i in employee_var if i not in y]

cols=['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 
      'department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low'] 

x = employee_data[cols]

y = employee_data['left']

from sklearn.model_selection  import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.28)
print(x_train)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(x_train, y_train)

rf.score(x_test,y_test)

# Saving model to disk
pickle.dump(rf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print("hello")
print(model.predict([[0.80, 0.86, 6,0,0,0,0,0,0,0]]))