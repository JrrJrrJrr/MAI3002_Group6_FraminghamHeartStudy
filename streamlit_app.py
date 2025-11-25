# Import convention
import streamlit as st

st.title("🎈 My new app")

st.write(
    "Streamlit Documentation: [docs.streamlit.io](https://docs.streamlit.io/)."
)

# Just add it after st.sidebar:
a = st.sidebar.radio('Choose:',[1,2])

