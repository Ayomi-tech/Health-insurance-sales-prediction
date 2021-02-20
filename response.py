#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import re
import pandas as pd


# In[2]:


def response():
    
    Age= input('Please enter your age - range of 20 and above:')
    while re.search('[0-9]',Age) is None:
        print('Make sure your age is in numbers') 
        Age= input('Please enter your age - range of 20 and above:')
    Annual_Premium = input('Please enter your desired Annual Premium - range of 2630 and more:')
    while re.search('[0-9]',Annual_Premium) is None:
        print('Make sure your Annual Premium is in numbers')
        Annual_Premium = input('Please enter your desired Annual Premium - range of 2630 and more:')
    Vintage = input('How many day have you been associated with our company - up to 300 days :')
    while re.search('[0-9]',Vintage) is None:
        print('Make sure your Vintage is in numbers')
        Vintage = input('How many day have you been associated with our company - up to 300 days :')

    Gender = input('Please enter your gender, Male=1, Female=0 :')
    while re.search('[0-1]',Gender) is None:
        print('Make sure your Gender is either 1 or 0')
        Gender = input('Please enter your gender, Male=1, Female=0 :')

    Driving_License = input('Do you have a Driving License? Yes=1, No=0 :')
    while re.search('[0-1]',Driving_License) is None:
        print('Make sure your Driving License either 1 or 0')
        Driving_License = input('Do you have a Driving License? Yes=1, No=0 :')

    Previously_Insured= input('Have you been previously insured with our company? Yes=1, No=0 :')
    while re.search('[0-1]',Previously_Insured) is None:
        print('Make sure that the value is either 1 or 0')
        Previously_Insured= input('Have you been previously insured with our company? Yes=1, No=0 :')

    Vehicle_Damage = input('Has your vehicle been damaged before? Yes=1, No=0 :')
    while re.search('[0-1]',Vehicle_Damage) is None:
        print('Make sure that your entry for vehicle damage is either 1 or 0')
        Vehicle_Damage = input('Has your vehicle been damaged before? Yes=1, No=0 :')

    VA_1_2_year = input('Is your vehicle 1-2 years old? Yes=1, No=0, if your answer is Yes, then please choose No in the next question:')
    while re.search('[0-1]',VA_1_2_year) is None:
        print('Make sure your entry is either 1 or 0')
        VA_1_2_year = input('Is your vehicle 1-2 years old? Yes=1, No=0, if your answer is Yes, then please choose No in the next question:')

    VA_below_1_year = input('Is your vehicle less than 1 year old? Yes=1, No=0: ')
    while re.search('[0-1]',VA_below_1_year) is None:
        print('Make sure your entry is either 1 or 0')
        VA_below_1_year = input('Is your vehicle less than 1 year old? Yes=1, No=0: ')

    #logistic_model_u=LogisticRegression().fit(X_train_under, y_train_under)
    loaded_model = pickle.load(open('my_model.pickle', 'rb'))
    parameters = ['Age', 'Annual_Premium', 'Vintage', 'Gender', 'Driving_License',
       'Previously_Insured', 'Vehicle_Damage', 'VA_1_2_year','VA_below_1_year']
    my_predictors = [Age, Annual_Premium, Vintage, Gender, Driving_License,
       Previously_Insured, Vehicle_Damage, VA_1_2_year ,VA_below_1_year]
    resp_data = dict(zip(parameters, my_predictors))
    resp_df = pd.DataFrame(resp_data, index=[0])
    #resp_y_pred= logistic_model_u.predict(resp_df)
    resp_y_pred= loaded_model.predict(resp_df)

    if resp_y_pred==1:
        return 'Hooray! This client is taking up our insurance plan'
    elif resp_y_pred==0:
        return 'Oops! This client is not taking up our insurance plan'
    #return f"Response: {float(resp_y_pred)}"


# In[3]:


print(response())


# In[ ]:




