# To run this code with streamlit server rin the following code
# streamlit run slit-test.py

import streamlit as st
import pandas as pd

@st.cache
def get_data():
    url = "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-09-12/visualisations/listings.csv"
    return pd.read_csv(url)

df = get_data()


st.title("Test for Streamlit")
st.markdown("Test is conducted under some requirements: ")

st.dataframe(df.head())


entities =  ["Mr",
 ".",
 "Trump",
 "’",
 "s",
 "tweets ",
 "began",
 "just ",
 "moments",
 "after",
 "a",
 "Fox",
 "News ",
 "report ",
 "by",
 "Mike ",
 "Tobin",
 ",",
 "a",
 "reporter",
 "for",
 "the",
 "network",
 ",",
 "about",
 "protests",
 "in",
 "Minnesota ",
 "and",
 "elsewhere ",
 ".",
 "India",
 "and",
 "China",
 "have ",
 "agreed ",
 "to",
 "peacefully"
 "resolve",
 "a",
 "simmering ",
 "border ",
 "dispute",
 "between",
 "the",
 "world",
 "'",
 "s",
 "two",
 "most",
 "populous",
 "nations",
 ",",
 "officials ",
 "in",
 "New",
 "Delhi",
 "said",
 ".",
  ]


labels =  ["O",
 "O",
 "B-PER",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "B-ORG",
 "B-ORG",
 "O",
 "O",
 "B-PER",
 "B-PER",
 "O",
 "O",
 "O",
 "O",
 "O",
 "B-ORG",
 "O",
 "O",
 "O",
 "O",
 "B-LOC",
 "O",
 "O",
 "O",
 "B-LOC",
 "O",
 "B-LOC",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "B-LOC",
 "B-LOC",
 "O",
 "O",
 ]

with st.echo():
    for e,l in zip(entities, labels):
        
        st.write("{:5}  {}".format(l, e))





















with st.echo():
    x = 20
    y = 20

with st.echo():
    z = x + y
    st.write(z)




