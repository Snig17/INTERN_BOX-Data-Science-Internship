#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import streamlit as st


# In[2]:


loaded_model = pickle.load(open('C:\\Users\\Snigdha\\New folder\\Studies\\Intern-Box\\Notes\\Tasks\\D-2\\train.sav', 'rb'))


# In[ ]:


def Churn_Prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The customer have stopped using your service'
    else:
      return 'The customer is using your service'
  
    
  
def main():
    
    
    # giving a title
    st.title('Churn prediction Web App')
    
    
    # getting the input data from the user
    
    
    AccountWeeks = st.text_input('Number of AccountWeeks')
    ContractRenewal = st.text_input('ContractRenewal')
    DataPlan = st.text_input('DataPlan')
    DataUsage= st.text_input('DataUsage')
    CustServCalls = st.text_input('CustServCalls')
    DayMins= st.text_input('No.of min in a day')
    DayCalls = st.text_input('No.of calls per day')
    MonthlyCharge = st.text_input('MonthlyCharge Value')
    OverageFee = st.text_input('OverageFee')
    RoamMins = st.text_input('RoamMins')
    
    
    
    # code for Prediction
    Churn = ''
    
    # creating a button for Prediction
    
    if st.button('Churn Prediction Result'):
        Churn = Churn_Prediction([AccountWeeks, ContractRenewal, DataPlan, DataUsage,
       CustServCalls, DayMins, DayCalls, MonthlyCharge, OverageFee,
       RoamMins])
        
        
    st.success(Churn)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    

