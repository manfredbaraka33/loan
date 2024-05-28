import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split





# Main Streamlit app
st.title('ElopyX finances')

st.write('Enter the details below to predict loan approval:')


# User input fields with explanations
gender = st.number_input("Select your gender('Male':1,'Female':0)",min_value=0,max_value=1)
married = st.number_input("Marital Status('No':0,'Yes':1)",min_value=0,max_value=1)
dependents = st.number_input('Dependents', min_value=0, help="Number of dependents (e.g., children or other dependents)")
education = st.number_input("Education('Graduate':1,'Not Graduate':0)",min_value=0,max_value=1)
self_employed = st.number_input('Self Employed("No:0,Yes:1")', help="Are you self-employed?",min_value=0,max_value=1)
applicant_income = st.number_input('Applicant Income in USD', help="Enter your monthly income in USD")
coapplicant_income = st.number_input('Coapplicant Income in USD', help="Enter co-applicant's monthly income in USD")
loan_amount = st.number_input('Loan Amount in USD', help="Enter the requested loan amount in USD")
loan_amount_term = st.number_input('Loan Amount Term in years', help="Enter the term of the loan in months",min_value=0)
credit_history = st.number_input("Credit History('No':0, 'Yes':1)",min_value=0,max_value=1 , help="Do you have a credit history?")
property_area = st.number_input("Property Area('Rural':0,'Semiurban':1,'Urban':2)", min_value=0,max_value=2, help="Select the area where you reside")

# Predict button
if st.button('Predict Loan Approval'):
    new_data = [[gender,married,dependents,education,self_employed,applicant_income,
                coapplicant_income,loan_amount,loan_amount_term,credit_history,property_area ]
 ]       
    
    # Load the loan dataset
    loan_dataset = pd.read_csv("loan.csv")
    # dropping the missing values
    loan_dataset = loan_dataset.dropna()
    
    # label encoding
    loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)
    
    # replacing the value of 3+ to 4
    loan_dataset = loan_dataset.replace(to_replace='3+', value=4)
    
    # convert categorical columns to numerical values
    loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
    
    
    # separating the data and label
    X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
    Y = loan_dataset['Loan_Status']
    
    
    X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)
    
    
    # Load the trained model
    model = LogisticRegression()
    model.fit(X_train,Y_train)
 
    prediction = model.predict(new_data)
    print(prediction)
    if prediction[0] == 1:
        loan_status='Approved' 
    elif prediction[0]==0:
        loan_status='Not Approved'
    else:
        loan_status='Something went wrong!'
  
    st.success(f'The loan status for the provided data is: {loan_status}')
