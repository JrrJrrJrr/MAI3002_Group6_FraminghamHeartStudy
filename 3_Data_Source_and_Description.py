import streamlit as st
import pandas as pd
from streamlit.components.v1 import html

st.set_page_config(
    page_title="Framingham Heart Study - Data Source",
    page_icon="❤️",
    #layout="wide"   
)
#st.markdown("""<style>body {zoom: 1;  /* Adjust this value as needed */}</style>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="font-size: 17px;">✍️ <strong>Authors (Group 6):</strong></div> 
\n&nbsp;                                  
<div style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Cleo Habets&nbsp;&nbsp;</div><br>              

<div style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Jerrica Pubben&nbsp;&nbsp;</div><br>              
                  
<div style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Noura al Sayed&nbsp;&nbsp;</div><br>
""", unsafe_allow_html=True)

st.sidebar.write("---")
st.sidebar.markdown("""📅 December 16th, 2025""")
st.sidebar.image("Ulogo.png")

############################# start page content #############################


st.title("Data Source and Description")
st.divider()

st.markdown("""
In our analysis, we utilized the **Framingham Heart Study** dataset. 
This longitudinal study tracks cardiovascular disease characteristics. 
The dataset contains information on **11,627** examinations from **4,434** unique participants 
over three different examination periods (visits).

It features **38 columns**, encompassing demographic details, risk factors (like smoking, diabetes), 
medical history, and outcome variables (CVD events). Detailed descriptions of the key columns are provided below.
""")

columns = {
    'Column Name': ['RANDID', 'SEX', 'TOTCHOL', 'AGE', 'SYSBP', 'DIABP', 'CURSOMKE', 'CIGPDAY', 
                    'BMI', 'DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE', 'PREVCHD', 'PREVAP', 'PREVMI', 
                    'PREVSTRK', 'PREVHYP', 'TIME', 'PERIOD', 'HDLC', 'LDLC', 'DEATH', 'ANGINA',
                    'HOSPMI', 'MI_FCHD', 'ANYCHD', 'STROKE', 'CVD', 'HYPERTEN', 'TIMEAP', 'TIMEMI',
                    'TIMEMIFC', 'TIMECHD', 'TIMESTRK', 'TIMECVD', 'TIMEDTH', 'TIMEHYP'],
    'Description': ['Unique identification number for each participant. Values range from 2448-999312.', 'Participant sex. 1 = Male (n = 5022), 2 = Female (n = 6605).', 'Serum Total Cholesterol (mg/dL). Values range from 107-696.',
                    'Age at exam (years). Values range from 32-81.', 'Systolic Blood Pressure (mean of last two of three measurements) (mmHg). Values range from 83.5-295. ', 'Diastolic Blood Pressure (mean of last two of three measurements) (mmHg). Values range from 30-150.',
                    'Current cigarette smoking at exam. 0 = Not current smoker (n = 6598), 1 = Current smoker (n = 5029).', 'Number of cigarettes smoked each day. 0 = Not current smoker. Values range from 0-90 cigarettes per day.', 'Body Mass Index, weight in kilograms/height meters squared. Values range from 14.43-56.8.',
                    'Diabetic according to criteria of first exam treated or first exam with casual glucose of 200 mg/dL or more. 0 = Not a diabetic (n = 11097), 1 = Diabetic (n = 530)', 'Use of Anti-hypertensive medication at exam. 0 = Not currently used (n = 10090), 1 = Current use (n = 944).', 'Heart rate (Ventricular rate) in beats/min. Values range from 37-220.',
                    'Casual serum glucose (mg/dL). Values range from 39-478.', 'Prevalent Coronary Heart Disease defined as pre-existing Angina Pectoris, Myocardial Infarction (hospitalized, silent or unrecognized), or Coronary Insufficiency (unstable angina). 0 = Free of disease (n = 10785), 1 = Prevalent disease (n = 842).',
                    'Prevalent Angina Pectoris at exam. 0 = Free of disease (n = 11000), 1 = Prevalent disease (n = 627).', 'Prevalent Myocardial Infarction. 0 = Free of disease (n = 11253), 1 = Prevalent disease (n = 374).', 'Prevalent Stroke. 0 = Free of disease (n = 11475), 1 = Prevalent disease (n = 152).',
                    'Prevalent Hypertensive. Subject was defined as hypertensive if treated or if second exam at which mean systolic was >=140 mmHg or mean Diastolic >=90 mmHg. 0 = Free of disease (n = 6283), 1 = Prevalent disease (n = 5344).', 'Number of days since baseline exam. Values range from 0-4854',
                    'Examination Cycle. 1 = Period 1 (n = 4434), 2 = Period 2 (n = 3930), 3 = Period 3 (n = 3263)', 'High Density Lipoprotein Cholesterol (mg/dL). Available for Period 3 only. Values range from 10-189.', 'Low Density Lipoprotein Cholesterol (mg/dL). Available for Period 3 only. Values range from 20-565.',
                    'Death from any cause. 0 = Did not occur during followup, 1 = Did occur during followup.', 'Angina Pectoris. 0 = Did not occur during followup, 1 = Did occur during followup.', 'Hospitalized Myocardial Infarction. 0 = Did not occur during followup, 1 = Did occur during followup.',
                    'Hospitalized Myocardial Infarction or Fatal Coronary Heart Disease. 0 = Did not occur during followup, 1 = Did occur during followup.', 'Angina Pectoris, Myocardial infarction (Hospitalized and silent or unrecognized), Coronary Insufficiency (Unstable Angina), or Fatal Coronary Heart Disease. 0 = Did not occur during followup, 1 = Did occur during followup.',
                    'Atherothrombotic infarction, Cerebral Embolism, Intracerebral Hemorrhage, or Subarachnoid Hemorrhage or Fatal Cerebrovascular Disease. 0 = Did not occur during followup, 1 = Did occur during followup.',  'Myocardial infarction (Hospitalized and silent or unrecognized), Fatal Coronary Heart Disease, Atherothrombotic infarction, Cerebral Embolism, Intracerebral Hemorrhage, or Subarachnoid Hemorrhage or Fatal Cerebrovascular Disease. 0 = Did not occur during followup, 1 = Did occur during followup.',
                    'Hypertensive. Defined as the first exam treated for high blood pressure or second exam in which either Systolic is 6 140 mmHg or Diastolic 6 90mmHg. 0 = Did not occur during followup, 1 = Did occur during followup.', 'Number of days from Baseline exam to first Angina during the followup or Number of days from Baseline to censor date. Censor date may be end of followup, death or last known contact date if subject is lost to followup.',
                    'Number of days from Baseline exam to first HOSPMI event during followup or Number of days from Baseline to censor date. Censor date may be end of followup, death or last known contact date if subject is lost to followup.', 'Number of days from Baseline exam to first MI_FCHD event during followup or Number of days from Baseline to censor date. Censor date may be end of followup, death or last known contact date if subject is lost to followup.',
                    'Number of days from Baseline exam to first ANYCHD event during followup or Number of days from Baseline to censor date. Censor date may be end of followup, death or last known contact date if subject is lost to followup.', 'Number of days from Baseline exam to first STROKE event during followup or Number of days from Baseline to censor date. Censor date may be end of followup, death or last known contact date if subject is lost to followup.',
                    'Number of days from Baseline exam to first CVD event during followup or Number of days from Baseline to censor date. Censor date may be end of followup, death or last known contact date if subject is lost to followup.',
                    'Number of days from Baseline exam to death if occurring during followup or Number of days from Baseline to censor date. Censor date may be end of followup, or last known contact date if subject is lost to followup.', 'Number of days from Baseline exam to first HYPERTEN event during followup or Number of days from Baseline to censor date. Censor date may be end of followup, death or last known contact date if subject is lost to followup.',
                    ]
    
}

with st.expander("👆 Click to see columns description"):
    columns_df = pd.DataFrame(columns)
    st.table(columns_df)

st.write("")
st.write("")

RL_DATASET = "https://github.com/LUCE-Blockchain/Databases-for-teaching/blob/main/Framingham%20Dataset.csv"

st.markdown(
    f'<a href="{URL_DATASET}" style="display: inline-block; padding: 12px 20px; background-color: #085492; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">**Click here to view raw dataset source**</a>',
    unsafe_allow_html=True
)

st.image("./cdc.png")
