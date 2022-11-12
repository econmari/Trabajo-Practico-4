#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


# In[ ]:


Changed_Credit_Limit_default = -6.48
Num_Credit_Inquiries_default = 0
Interest_Rate_default = 1
Outstanding_Debt_default = 0.23
Credit_Mix_default = 1


# In[ ]:


st.title('Credit Score Classifier')
st.caption('Made by: Marisol Añon, Micaela Vittor, Gonzalo Cardozo, Flor Horno, Sofia Smith')


# In[ ]:


with st.sidebar:
    st.header('Credit Score Form')
    Credit_Mix = st.number_input('Predominant Credit:', min_value=1, max_value=3, value=Credit_Mix_default)
    Outstanding_Debt = st.number_input('Remaining debt to be paid:', min_value=0.23, max_value=4998.07, step=0.01, value=Outstanding_Debt_default)
    Interest_Rate = st.number_input('Interest rate on credit card', min_value=1, max_value=34, step=1, value=Interest_Rate_default)
    Num_Credit_Inquiries = st.number_input('Number of Credit Inquiries', min_value=0, max_value=20, step=1, value=Num_Credit_Inquiries_default)
    Changed_Credit_Limit = st.number_input('Percentage change in credit card limit', min_value=-6.48, max_value=36.29, step=0.01, value=Changed_Credit_Limit_default)
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
        X = pd.DataFrame([[Credit_Mix, Outstanding_Debt, Interest_Rate, Num_Credit_Inquiries, Changed_Credit_Limit]], 
                         columns = ["Credit_Mix", "Outstanding_Debt", "Interest_Rate", "Num_Credit_Inquiries","Changed_Credit_Limit"])

        # Get prediction
        credit_score = model.predict(X)[0]

        if credit_score == 1:
            st.balloons()
            t1 = plt.Polygon([[5, 0.5], [5.5, 0], [4.5, 0]], color='black')
            placeholder.markdown('Your credit score is **GOOD**! Congratulations!')
            st.markdown('This credit score indicates that this person is likely to repay a loan, so the risk of giving them credit is low.')
        elif credit_score == 2:
            t1 = plt.Polygon([[3, 0.5], [3.5, 0], [2.5, 0]], color='black')
            placeholder.markdown('Your credit score is **STANDARD**.')
            st.markdown('This credit score indicates that this person is likely to repay a loan, but can occasionally miss some payments. Meaning that the risk of giving them credit is medium.')
        elif credit_score == 3:
            t1 = plt.Polygon([[1, 0.5], [1.5, 0], [0.5, 0]], color='black')
            placeholder.markdown('Your credit score is **POOR**.')
            st.markdown('This credit score indicates that this person is unlikely to repay a loan, so the risk of lending them credit is high.')
        plt.gca().add_patch(t1)
        figure.pyplot(f)
        prob_fig, ax = plt.subplots()

        with st.expander('Click to see how certain the algorithm was'):
            plt.pie(model.predict_proba(X)[0], labels=['Poor', 'Standard', 'Good'], autopct='%.0f%%')
            st.pyplot(prob_fig)

        with st.expander('Click to see how much each feature weight'):
            importance = model.feature_importances_
            importance = pd.DataFrame(importance)
            columns = pd.DataFrame(['Credit_Score', 'Credit_Mix', 'Outstanding_Debt', 'Interest_Rate','Num_Credit_Inquiries', 'Changed_Credit_Limit'])

            importance = pd.concat([importance, columns], axis=1)
            importance.columns = ['importance', 'index']
            importance_fig.sort_values(by='importance', ascending=True, inplace=True)

            # plotting the figure
            importance_figure, ax = plt.subplots()
            bars = ax.barh('index', 'importance', data=importance_fig)
            ax.bar_label(bars)
            plt.ylabel('')
            plt.xlabel('')
            plt.xlim(0,20)
            sns.despine(right=True, top=True)
            st.pyplot(importance_figure)
# In[ ]:




