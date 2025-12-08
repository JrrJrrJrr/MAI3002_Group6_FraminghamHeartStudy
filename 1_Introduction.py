import streamlit as st
import base64

st.set_page_config(
    page_title="Cardiovascular disease - Introduction",
    page_icon="📊",
    #layout="wide"   
)
#st.markdown("""<style>body {zoom: 1.4;  /* Adjust this value as needed */}</style>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="font-size: 17px;">✍️ <strong>Authors:</strong></div> 
\n&nbsp;                                  
<a href="https://www.linkedin.com/in/amralshatnawi/" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Amr Alshatnawi&nbsp;&nbsp;</a><br>             

<a href="https://www.linkedin.com/in/hailey-pangburn" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Hailey Pangburn&nbsp;&nbsp;</a><br>             
                    
<a href="mailto:mcmasters@uchicago.edu" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">Richard McMasters</a><br>
""", unsafe_allow_html=True)

st.sidebar.write("---")
st.sidebar.markdown("""📅 March 9th, 2024""")

st.sidebar.image("Ulogo.png")

# def add_side_title():
#     st.markdown(
#         """
#         <style>
#             [data-testid="stSidebarNav"]::before {
#                 content:"MSBI 32000 Winter 2024";
#                 margin-left: 20px;
#                 margin-top: 20px;
#                 font-size: 25px;
#                 position: relative;
#                 top: 80px;
#             }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

# add_side_title()

############################# start page content #############################

st.title("Introduction")
st.divider()

st.markdown("""
Cardiovascular disease (CVD) is a leading cause of death worldwide and is closely linked to arterial stiffness. Pulse pressure (PP), defined as systolic minus diastolic blood pressure, is a well-established marker of arterial stiffness, with higher values indicating increased CVD risk. The Framingham Heart Study, with repeated blood pressure measurements and longitudinal follow-up of cardiovascular events, provides an ideal setting to examine whether changes in pulse pressure over time predict future CVD.

In this study, we analyzed cleaned Framingham data (11,627 observations), reshaped into one row per participant (n = 3,206) with three complete visits. Pulse pressure was calculated (SYSBP − DIABP), and change between Visit 1 and Visit 2 (ΔPP) was used as the predictor, while CVD occurrence at Visit 3 served as the outcome.
""")

#st.image("./covid.png")



def get_image_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Path to image
image_path = "./covid.png"

# Convert your image file to a base64 string
image_base64 = get_image_base64(image_path)
image_html = f'<img src="data:image/png;base64,{image_base64}" class="custom-img">'

st.image("./covid.png")

