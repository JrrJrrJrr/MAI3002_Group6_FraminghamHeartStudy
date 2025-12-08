# pages/researchquestion.py
import streamlit as st

st.set_page_config(
    page_title="Cardiovascular Disease - Research Questions",
    page_icon="Question",
    layout="centered"
)

# =============================================
# SIDEBAR – MOOIE RODE BLOKJES (perfect werkend!)
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
st.title("Research Questions")
st.divider()

st.subheader("1. Can changes in pulse pressure between Visit 1 and Visit 2 predict the occurrence of a CVD event before Visit 3 in Framingham participants?")

st.markdown("""
- **H₀**: There is no significant association between ΔPP and CVD occurrence.  
- **H₁**: An increase in pulse pressure (ΔPP) significantly predicts CVD before Visit 3.
""")

st.subheader("2. Is the association between ΔPP and CVD different for women vs men?")

st.markdown("""
- **H₀**: The relationship between ΔPP and CVD risk is the same in men and women.  
- **H₁**: The relationship between ΔPP and CVD risk differs between men and women.
""")

st.markdown("---")
st.caption("MAI3002 – Framingham Heart Study | 2025 | Cleo • Noura • Jerr")



