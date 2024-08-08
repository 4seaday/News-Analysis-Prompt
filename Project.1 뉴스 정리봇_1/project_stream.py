import streamlit as st
import pandas as pd

from pymongo import MongoClient

client = MongoClient(host="localhost", port=27017)
db = client["project1"]
collection = db["NewsAnalysis"]

for item in collection.find():
    print(item)

# 샘플 데이터
data = [
    {"organization": "기업 A", "date": "2024-04-20", "positive": 0.6, "negative": 0.2, "neutral": 0.2},
    {"organization": "기업 A", "date": "2024-04-21", "positive": 0.5, "negative": 0.3, "neutral": 0.2},
    {"organization": "기업 A", "date": "2024-04-22", "positive": 0.7, "negative": 0.1, "neutral": 0.2},
    {"organization": "기업 B", "date": "2024-04-20", "positive": 0.4, "negative": 0.4, "neutral": 0.2},
    {"organization": "기업 B", "date": "2024-04-21", "positive": 0.5, "negative": 0.3, "neutral": 0.2},
    {"organization": "기업 B", "date": "2024-04-22", "positive": 0.6, "negative": 0.2, "neutral": 0.2},
    {"organization": "기업 C", "date": "2024-04-20", "positive": 0.7, "negative": 0.1, "neutral": 0.2},
    {"organization": "기업 C", "date": "2024-04-21", "positive": 0.8, "negative": 0.1, "neutral": 0.1},
    {"organization": "기업 C", "date": "2024-04-22", "positive": 0.9, "negative": 0.05, "neutral": 0.05},
]

# DataFrame으로 변환
df = pd.DataFrame(data)

# 날짜 형식 변환
df['date'] = pd.to_datetime(df['date'])

# Streamlit title
st.title("기업별 날짜에 따른 감성 스코어 변화")

# 기업명 선택
organization = st.selectbox("기업을 선택하세요", df['organization'].unique())

# 선택한 기업의 데이터 필터링
selected_data = df[df['organization'] == organization].set_index('date')

# 감성 스코어 라인차트 그리기
st.line_chart(selected_data[['positive', 'negative', 'neutral']])
