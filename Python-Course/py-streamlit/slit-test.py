# To run this code with streamlit server rin the following code
# streamlit run slit-test.py

import streamlit as st

st.title("Test rizvi")

with st.echo():
    x = 20
    y = 20

with st.echo():
    z = x + y
    st.write(z)


