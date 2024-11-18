import streamlit as st
import eda
import prediction

page = st.sidebar.selectbox('Pilih Halaman: ', ('EDA', 'prediction'))

if page == 'EDA':
    eda.run()
else:
    prediction.run()