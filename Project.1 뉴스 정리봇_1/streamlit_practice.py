import streamlit as st

st.title("첫 번째 데모입니다.")
st.header("이것은 header입니다.")
st.subheader("이것은 subheader입니다.")
st.text("이것은 텍스트입니다.")
st.markdown("***마크다운 문법 적용확인***")
st.code('print("Hello, streamlit!")', language='python')

# 입력 컴포넌트
name = st.text_input("이름을 입력하세요: ")
age = st.number_input("나이를 입력하시오: ", min_value=0, max_value=120)

st.write("이름: ", name)
st.write("나이: ", age)

# 버튼

if st.button("확인 버튼."):
    st.write("확인되셨습니다.")
else:
    st.write("취소되셨습니다.")