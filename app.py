import streamlit as st

st.image('dell_picture.jpg')
st.title("Testing")

st.date_input("Transaction Date")
st.radio("Your department:",['A','B','C','D'])

st.camera_input("Case reported")
