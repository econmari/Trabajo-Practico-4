#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler


# In[ ]:


st.title('Credit Score Classifier')
st.caption('Made by: Marisol Añon, Micaela Vittor, Gonzalo Cardozo, Flor Horno, Sofia Smith')


# In[ ]:


with st.sidebar:
    st.header('Credit Score Form')
    Credit_Mix = st.selectbox('Credit Mix:', start= 1,end=3, step=1)
    Outstanding_Debt = st.number_input('Remaining debt to be paid')
    Interest_Rate = st.number_input('Interest rate on credit card')
    Changed_Credit_Limit = st.number_input('Percentage change in credit card limit')
    Total_Months_Credit_History_Age=  st.number_input('Total months of credit history', step = 1)
    Num_Credit_Card = st.number_input('Number of other credit cards held by the customer', step= 1)
    Delay_from_due_date = st.number_input("Average number of days delayed from the payment date", step=1)
    Num_of_Loan= st.number_input("Number of loans taken from the bank", step=1)
    Num_Bank_Accounts= st.number_input("Number of bank accounts held by the customer", step=1)
    Monthly_Balance = st.number_input("Monthly balance amount of the customer")
    run = st.button( 'Run the numbers!')

# In[ ]:


st.header('Credit Score Results')


# In[ ]:


col1, col2 = st.columns([3, 2])

with col2:
    x1 = [0, 6, 0]
    x2 = [0, 4, 0]
    x3 = [0, 2, 0]
    y = ['0', '1', '2']

    f, ax = plt.subplots(figsize=(5,2))

    p1 = sns.barplot(x=x1, y=y, color='#3EC300')
    p1.set(xticklabels=[], yticklabels=[])
    p1.tick_params(bottom=False, left=False)
    p2 = sns.barplot(x=x2, y=y, color='#FAA300')
    p2.set(xticklabels=[], yticklabels=[])
    p2.tick_params(bottom=False, left=False)
    p3 = sns.barplot(x=x3, y=y, color='#FF331F')
    p3.set(xticklabels=[], yticklabels=[])
    p3.tick_params(bottom=False, left=False)

    plt.text(0.7, 1.05, "POOR", horizontalalignment='left', size='medium', color='white', weight='semibold')
    plt.text(2.5, 1.05, "STANDARD", horizontalalignment='left', size='medium', color='white', weight='semibold')
    plt.text(4.7, 1.05, "GOOD", horizontalalignment='left', size='medium', color='white', weight='semibold')

    ax.set(xlim=(0, 6))
    sns.despine(left=True, bottom=True)

    figure = st.pyplot(f, width=5)


# In[ ]:


with col1:

    placeholder = st.empty()

    if run:
        
        # Unpickle classifier
        model = joblib.load("model.pkl")

        # Store inputs into dataframe

        #if Credit_Mix == 'Negative':
        #    Negative = 1
        #elif Credit_Mix == 'Neutral':
        #    Neutral = 2
        #else:
        #    Positive = 3
            
        scaler = StandardScaler()
        X =  pd.DataFrame(scaler.fit_transform([[Credit_Mix, Outstanding_Debt, Interest_Rate, Changed_Credit_Limit, Total_Months_Credit_History_Age, Num_Credit_Card, Delay_from_due_date, Num_of_Loan, Num_Bank_Accounts, Monthly_Balance]], columns = ["Credit_Mix", "Outstanding_Debt", "Interest_Rate", "Changed_Credit_Limit", "Total_Months_Credit_History_Age", "Num_Credit_Card", "Delay_from_due_date", "Num_of_Loan", "Num_Bank_Accounts", "Monthly_Balance"])
        
        #X = pd.DataFrame([[Credit_Mix, Outstanding_Debt, Interest_Rate, Changed_Credit_Limit, Total_Months_Credit_History_Age, Num_Credit_Card, Delay_from_due_date, Num_of_Loan, Num_Bank_Accounts, Monthly_Balance]], 
        #                 columns = ["Credit_Mix", "Outstanding_Debt", "Interest_Rate", "Changed_Credit_Limit", "Total_Months_Credit_History_Age", "Num_Credit_Card", "Delay_from_due_date", "Num_of_Loan", "Num_Bank_Accounts", "Monthly_Balance"])
        #X = X.replace(["Negative", "Neutral", "Positive"], [1, 2, 3])
                          
        # Get prediction
        credit_score = model.predict(X)[0]

        if credit_score == 0:
            st.balloons()
            t1 = plt.Polygon([[5, 0.5], [5.5, 0], [4.5, 0]], color='black')
            placeholder.markdown('Your credit score is **GOOD**! Congratulations!')
            st.markdown('This credit score indicates that this person is likely to repay a loan, so the risk of giving them credit is low.')
        elif credit_score == 2:
            t1 = plt.Polygon([[3, 0.5], [3.5, 0], [2.5, 0]], color='black')
            placeholder.markdown('Your credit score is **STANDARD**.')
            st.markdown('This credit score indicates that this person is likely to repay a loan, but can occasionally miss some payments. Meaning that the risk of giving them credit is medium.')
        elif credit_score == 1:
            t1 = plt.Polygon([[1, 0.5], [1.5, 0], [0.5, 0]], color='black')
            placeholder.markdown('Your credit score is **POOR**.')
            st.markdown('This credit score indicates that this person is unlikely to repay a loan, so the risk of lending them credit is high.')
        plt.gca().add_patch(t1)
        figure.pyplot(f)
        prob_fig, ax = plt.subplots()

        with st.expander('Click to see how certain the algorithm was'):
            plt.pie(model.predict_proba(X)[0], labels=['Good', 'Poor', 'Standard'], autopct='%.0f%%')
            st.pyplot(prob_fig)
        




