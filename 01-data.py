import streamlit as st
import pandas as pd
import numpy as np

st.title("데이터프레임 튜토리얼")

dataframe = pd.DataFrame(
    {'first_column': [1, 2, 3, 4],
     'second_column': [10, 20, 30, 40]
    }
)

st.dataframe(dataframe, use_container_width=False)

st.table(dataframe)

st.metric(label='온도', value='10도', delta='1.2')
st.metric(label='삼성전자', value='61,000원', delta = '-1,200원')

col1, col2, col3 = st.columns(3)
col1.metric(label='온도', value='10도', delta='1.2')
col2.metric(label='삼성전자', value='61,000원', delta = '-1,200원')
col3.metric(label='유럽', value='1,335원', delta = '11.44원')