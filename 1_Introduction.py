# pages/intro.py
import streamlit as st
import base64

st.set_page_config(
    page_title="Cardiovascular Disease - Introduction",
    page_icon="Chart",
    layout="centered"
)

# =============================================
# SIDEBAR – PRECIES ZOALS OP JE RESEARCH QUESTION PAGINA
# =============================================
st.sidebar.markdown("""
<div style="font-size: 18px; font-weight: 600; margin-bottom: 16px;">
    Authors:
</div>

<div style="background-color: #871212; color: white; padding: 12px 20px; border-radius: 10px; text-align: center; font-size: 16px; font-weight: 500; margin: 10px 0;">
    Cleo
</div>

<div style="background-color: #871212; color: white; padding: 12px 20px; border-radius: 10px; text-align: center; font-size: 16px; font-weight: 500; margin: 10px 0;">
    Noura
</div>

<div style="background-color: #871212; color: white; padding: 12px 20px; border-radius: 10px; text-align: center; font-size: 16px; font-weight: 500; margin: 10px 0;">
    Jerr
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Date:** 2025")
st.sidebar.markdown("**Course:** MAI3002")

try:
    st.sidebar.image("Ulogo.png", use_column_width=True)
except:
    pass

# =============================================
# HOOFDINHOUD
# =============================================
st.title("Introduction")
st.divider()

st.markdown("""
Cardiovascular disease (CVD) is a leading cause of death worldwide and is closely linked to arterial stiffness.  
**Pulse pressure (PP)**, defined as systolic minus diastolic blood pressure, is a well-established marker of arterial stiffness, with higher values indicating increased CVD risk.

The **Framingham Heart Study**, with repeated blood pressure measurements and longitudinal follow-up of cardiovascular events, provides an ideal setting to examine whether changes in pulse pressure over time predict future CVD.

In this study, we analyzed cleaned Framingham data (**11,627 observations**), reshaped into one row per participant (**n = 3,206**) with three complete visits.  
Pulse pressure was calculated (**SYSBP − DIABP**), and the change between Visit 1 and Visit 2 (**ΔPP**) was used as predictor for CVD occurrence at Visit 3.
""")

try:
    st.image("covid.png", caption="Pulse Pressure and CVD Risk Progression", use_column_width=True)
except:
    st.info("Afbeelding 'covid.png' niet gevonden")

st.markdown("---")
st.caption("MAI3002 – Framingham Heart Study | 2025 | Cleo • Noura • Jerr")

