import streamlit as st
import pandas as pd
from datetime import datetime as dt
import datetime

button = st.button("버튼을 눌러보세요")

if button:
    st.write(":blue[버튼]이 눌렸습니다")

dataframe = pd.DataFrame(
    {"first_column": [1, 2, 3, 4],
     "second_column": [10, 20, 30, 40]
    }
)

st.download_button(
    label = 'csv로 다운로드',
    data = dataframe.to_csv(),
    file_name='sample.csv',
    mime ='text/csv'
)

agree = st.checkbox("동의 하십니까")

if agree:
    st.write('동의해주셔서 감사합니다 :100:') 

mbti = st.radio(
    '당신의 mbti는 무엇입니까?',
    ('infp', 'enfp', '선택지 없음')
)

if mbti == 'infp':
    st.write('연준 튜터와 같네요!')
elif mbti == 'enfp':
    st.write('연준 튜터와 다르네요!')
else:
    st.write('당신의 mbti를 알고싶어요')

mbti = st.selectbox(
    '당신의 mbti는 무엇입니까?',
    ('infp', 'enfp', '선택지 없음'),
    index = 2
)

if mbti == 'infp':
    st.write('연준 튜터와 같네요!')
elif mbti == 'enfp':
    st.write('연준 튜터와 다르네요!')
else:
    st.write('당신의 mbti를 알고싶어요')    


option = st.multiselect(
    '당신이 좋아하는 과일은 뭔가요?',
    ['망고', '오렌지', '사과', '바나나'],
    ['망고', '오렌지']
)

st.write(f'당신의 선택은 :red[{option}] 입니다')

value = st.slider(
    '범위의 값을 다음과 같이 지정할 수 있어요',
    0.0, 100.0, (25.0, 75.0)
)

st.write('선택 범위', value)

text = st.text_input(
    label = '가고 싶은 여행지가 있나요?',
    placeholder = '여행지를 입력해 주세요'
)

st.write(f'당신이 선택한 여행지: :violet[{text}]')

number = st.number_input(
    label = '나이를 입력해 주세요',
    min_value = 10,
    max_value = 100,
    value = 30,
    step = 5
)

st.write('당신의 나이는 : ', number)