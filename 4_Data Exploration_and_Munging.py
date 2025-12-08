import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Framingham Heart Study - Data Exploration, Cleaning, and Feature Engineering",
    page_icon="❤️",
    layout="wide"
)

# Sidebar Authors
st.sidebar.markdown("""<div style="font-size: 17px;">✍️ <strong>Authors (Group 6):</strong></div> 
\n&nbsp;                                  
<div style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Cleo Habets&nbsp;&nbsp;</div><br>              

<div style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Jerrica Pubben&nbsp;&nbsp;</div><br>              
                  
<div style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Noura al Sayed&nbsp;&nbsp;</div><br>
""", unsafe_allow_html=True)

st.sidebar.write("---")
st.sidebar.markdown("""📅 December 16th, 2025""")

############################# Start Page Content #############################

st.title("Data Exploration and Munging")
st.divider()

# -----------------------------------------------------------------------------
# 1. Data Exploration
# -----------------------------------------------------------------------------
st.header("1. Data Exploration")

# Load Data
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv'
    return pd.read_csv(url)

data = load_data()

st.markdown("We begin by loading the raw longitudinal dataset.")
st.write(f"The dataset contains **{data.shape[0]}** rows and **{data.shape[1]}** columns.")
st.write("Data preview:")
st.dataframe(data.head())

with st.expander("👆 Expand to view data loading code"):
    st.code("""
import pandas as pd

# Load dataset
url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv'
cvd = pd.read_csv(url)

print("Raw data shape:", cvd.shape) # Rows, columns
cvd.head()
    """)

st.subheader("Statistical Summary")
st.write(data.describe().T)

# -----------------------------------------------------------------------------
# 2. Longitudinal Structure
# -----------------------------------------------------------------------------
st.header("2. Longitudinal structure: periods and participants")

st.markdown("""
The Framingham Heart Study is longitudinal. Participants attend multiple examinations (Visits).
Here we analyze the distribution of records across periods and participants.
""")

# Metrics
col1, col2 = st.columns(2)
n_participants = data['RANDID'].nunique()
visit_counts = data.groupby('RANDID')['PERIOD'].nunique()

col1.metric("Unique Participants", n_participants)
col1.write("Distribution of visits per participant:")
col1.write(visit_counts.value_counts().sort_index())

with col2:
    st.write("**Records per Period (Visit Number):**")
    st.bar_chart(data['PERIOD'].value_counts().sort_index())

with st.expander("👆 Expand to view longitudinal analysis code"):
    st.code("""
# Period distribution
print("PERIOD distribution (rows):")
print(cvd['PERIOD'].value_counts().sort_index())

# 1 participant can have multiple rows (max. 3)
n_participants = cvd['RANDID'].nunique()
visit_counts = cvd.groupby('RANDID')['PERIOD'].nunique()

print(f"Unique participants: {n_participants}")
print("Visits per participant (value counts):")
print(visit_counts.value_counts().sort_index())
    """)

st.subheader("Interactive Participant Timeline")
st.markdown("Select a participant ID to view their health trajectory across the 3 visits.")

# Interactive Timeline Widget
selected_id = st.selectbox("Choose Participant ID (RANDID):", sorted(data['RANDID'].unique()))

user_data = data[data['RANDID'] == selected_id].sort_values('PERIOD')

if not user_data.empty:
    fig_timeline, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # BP
    axes[0].plot(user_data['PERIOD'], user_data['SYSBP'], marker='o', label='SYSBP', color='red')
    axes[0].plot(user_data['PERIOD'], user_data['DIABP'], marker='o', label='DIABP', color='blue')
    axes[0].set_title("Blood Pressure")
    axes[0].set_ylabel("mmHg")
    axes[0].set_xticks([1, 2, 3])
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # BMI
    axes[1].plot(user_data['PERIOD'], user_data['BMI'], marker='s', color='green', linestyle='--')
    axes[1].set_title("BMI")
    axes[1].set_ylabel("Kg/m²")
    axes[1].set_xticks([1, 2, 3])
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # Cholesterol
    axes[2].plot(user_data['PERIOD'], user_data['TOTCHOL'], marker='^', color='orange')
    axes[2].set_title("Cholesterol")
    axes[2].set_ylabel("mg/dL")
    axes[2].set_xticks([1, 2, 3])
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig_timeline)
else:
    st.write("No data found for this ID.")

# -----------------------------------------------------------------------------
# 3. Missing Data Overview
# -----------------------------------------------------------------------------
st.header("3. Missing data overview and analysis")

missing = data.isna().sum().sort_values(ascending=False)
missing_pct = (missing / len(data) * 100).round(2)
missing_df = pd.DataFrame({'Missing Values': missing, '%': missing_pct})
missing_df = missing_df[missing_df['Missing Values'] > 0]

st.dataframe(missing_df)

# Plotly Bar Chart for Missing Data
fig_missing = px.bar(missing_df, x='%', y=missing_df.index, orientation='h', 
                     title="Missing Values by Variable (%)",
                     labels={'index': 'Variable', '%': 'Percentage Missing'},
                     color='%', color_continuous_scale='Teal')
st.plotly_chart(fig_missing)

st.markdown("""
**Interpretation:**
* **LDLC & HDLC (~74% missing):** These were likely only measured in Period 3 (MNAR/Structural).
* **GLUCOSE (~12%):** Likely MAR (lab/patient factors).
* **Others (<5%):** Likely MCAR (random entry errors).
""")

with st.expander("👆 Expand to view missing data code"):
    st.code("""
# Overall missingness
missing = cvd.isna().sum().sort_values(ascending=False)
missing_pct = (missing / len(cvd) * 100).round(2)
missing_summary = pd.DataFrame({'Missing values': missing, '%': missing_pct})
missing_summary[missing_summary['Missing values'] > 0]
    """)

# -----------------------------------------------------------------------------
# 4. Distributions, Outliers & Feature Engineering
# -----------------------------------------------------------------------------
st.header("4. Distributions, outliers, data cleaning, feature engineering")

st.subheader("Numeric Distributions")
numeric_cols = ['SYSBP', 'DIABP', 'BMI', 'TOTCHOL', 'GLUCOSE', 'HEARTRTE', 'CIGPDAY']
selected_metric = st.selectbox("Select variable to visualize distribution:", numeric_cols)

# Histogram
fig_dist = px.histogram(data, x=selected_metric, nbins=30, title=f"Distribution of {selected_metric}",
                        color_discrete_sequence=['#1f77b4'])
st.plotly_chart(fig_dist)

st.subheader("Outlier Detection (Boxplots)")
# Boxplot
fig_box = px.box(data, y=selected_metric, title=f"Boxplot of {selected_metric}", points="outliers")
st.plotly_chart(fig_box)

with st.expander("👆 Expand to view distribution code"):
    st.code("""
# Using Plotly for interactive distributions
fig = px.histogram(cvd, x=selected_variable, title=f"Distribution of {selected_variable}")
fig.show()

# Boxplots for outliers
fig_box = px.box(cvd, y=selected_variable, title=f"Boxplot of {selected_variable}")
fig_box.show()
    """)

st.subheader("Feature Engineering: Pulse Pressure (PP)")
st.markdown("""
We create a new variable **Pulse Pressure (PP)** representing arterial stiffness.
$$ PP = SYSBP - DIABP $$
We are specifically interested in the change in PP ($\Delta PP$) between Visit 1 and Visit 2.
""")

# Calculate PP and Delta PP (Simplified for Demo)
df_clean = data.copy()
df_clean['PP'] = df_clean['SYSBP'] - df_clean['DIABP']

# Pivot to wide format to calculate delta
pp_wide = df_clean.pivot_table(index='RANDID', columns='PERIOD', values='PP').reset_index()
pp_wide.columns = ['RANDID', 'PP1', 'PP2', 'PP3']

# Calculate Delta
pp_wide['DELTA_PP'] = pp_wide['PP2'] - pp_wide['PP1']
valid_delta = pp_wide.dropna(subset=['DELTA_PP'])

st.write(f"Participants with valid PP at Visit 1 & 2: **{valid_delta.shape[0]}**")
st.write(valid_delta[['PP1', 'PP2', 'DELTA_PP']].describe().T)

with st.expander("👆 Expand to view Feature Engineering code"):
    st.code("""
# Compute ΔPP = PP2 − PP1, keep only participants with both visits
pp_delta = (
    pp_wide[['PP1', 'PP2']]
    .dropna()  # require both V1 & V2
    .assign(DELTA_PP=lambda d: d['PP2'] - d['PP1'])
    .reset_index()
)

print("Participants with PP1 & PP2:", pp_delta['RANDID'].nunique())
pp_delta[['DELTA_PP']].describe()
    """)

# -----------------------------------------------------------------------------
# 5. Analytic Dataset: Overview
# -----------------------------------------------------------------------------
st.header("5. Analytic dataset: overview")

st.markdown("""
The final analytic dataset consists of participants who meet the inclusion criteria:
1.  Complete Blood Pressure data at Visits 1 & 2 (to compute $\Delta PP$).
2.  Observed CVD outcome by Visit 3.
""")

st.markdown("""
| **Study stage / Inclusion criteria** | **Count (N)** |
| ---------------------------------------- | ------------- |
| Total participants                       | 4,434         |
| Participants with PP data at V1 & V2     | 3,930         |
| Participants with CVD data at V3         | 3,263         |
| **Final analytic sample ($\Delta PP$ & V3 outcome)** | **3,206** |
""")

# Creating the final analytic table (1 row pp)
# 1. Baseline covariates from Visit 1
baseline_v1 = data[data['PERIOD'] == 1][['RANDID', 'AGE', 'SEX', 'BMI', 'SYSBP', 'DIABP', 'GLUCOSE', 'TOTCHOL', 'CIGPDAY']].copy()
baseline_v1 = baseline_v1.add_prefix('V1_').rename(columns={'V1_RANDID': 'RANDID'})

# 2. Outcome from Visit 3
outcome_v3 = data[data['PERIOD'] == 3][['RANDID', 'CVD']].copy()

# 3. Delta PP (Calculated in the previous section)
pp_wide = data.pivot_table(index='RANDID', columns='PERIOD', values='SYSBP').reset_index() # Simplified for demo
# (In your real app, make sure pp_delta comes from your properly calculated PP dataframe)
# Using the mock valid_delta from previous steps:
pp_delta = valid_delta[['RANDID', 'DELTA_PP']]

st.markdown("### 7. Final analytic dataset (1 row per person)")

# Show the code used for merging
st.code("""
# 7. Final analytic dataset (1 row per person)
analytic = (
    pp_delta
      .merge(outcome_v3, on='RANDID', how='inner')   # require CVD outcome at Visit 3
      .merge(baseline_v1, on='RANDID', how='left')   # add baseline features
)

print("\\nFINAL analytic dataset shape:", analytic.shape)
analytic.head()
""", language='python')

# Perform the actual merge in the app
analytic = (
    pp_delta
      .merge(outcome_v3, on='RANDID', how='inner')
      .merge(baseline_v1, on='RANDID', how='left')
)

# Display the print output
st.text(f"FINAL analytic dataset shape: {analytic.shape}")

# Display the table 
st.dataframe(analytic.head())

# Interactive Plotting Widget
var_to_plot = st.selectbox("Select Variable:", ['DELTA_PP', 'V1_AGE', 'V1_BMI', 'V1_SYSBP', 'V1_TOTCHOL'])
split_by = st.radio("Split by:", ["CVD", "V1_SEX"], horizontal=True)

# Plotting logic
fig_comp = px.box(analytic, x=split_by, y=var_to_plot, 
                  color=split_by, 
                  title=f"Distribution of {var_to_plot} by {split_by}",
                  labels={"CVD": "CVD Event (0=No, 1=Yes)", "V1_SEX": "Sex (1=Male, 2=Female)"})
st.plotly_chart(fig_comp)

st.subheader("Machine Learning Setup")
st.markdown("We defined the following features for our predictive models:")
feature_cols = [
    "DELTA_PP", "V1_AGE", "V1_SEX", "V1_BMI", 
    "V1_SYSBP", "V1_DIABP", "V1_GLUCOSE", 
    "V1_TOTCHOL", "V1_CIGPDAY"
]
st.code(f"feature_cols = {feature_cols}", language="python")

st.markdown("""
The data was split into training and testing sets (stratified by CVD outcome) to account for class imbalance.
""")
