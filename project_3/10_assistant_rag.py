from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
import config
from openai import OpenAI
from pykrx import stock
from datetime import datetime

client = OpenAI(api_key=config.OPENAI_API_KEY)
model="gpt-4o"

data = [
    {
        "기업명": "삼성전자",
        "날짜": "2024-03-02",
        "문서 카테고리": "인수합병",
        "요약": "삼성전자가 HVAC(냉난방공조) 사업 인수를 타진 중이며, 이는 기존 가전 사업의 약점 보완을 목적으로 한다.",
        "주요 이벤트": ["기업 인수합병"]
    },
    {
        "기업명": "삼성전자",
        "날짜": "2024-03-24",
        "문서 카테고리": "인수합병",
        "요약": "테스트 하나 둘 셋",
        "주요 이벤트": ["신제품 출시"]
    },
    {
        "기업명": "현대차",
        "날짜": "2024-04-02",
        "문서 카테고리": "인수합병",
        "요약": "삼성전자가 HVAC(냉난방공조) 사업 인수를 타진 중이며, 이는 기존 가전 사업의 약점 보완을 목적으로 한다.",
        "주요 이벤트": ["기업 인수합병", "신제품 출시"]
    },
]

doc_list = [item['요약'] for item in data]

bm25_retriever = BM25Retriever.from_texts(
    doc_list, metadatas=[{"source": i} for i in range(len(data))]
)
bm25_retriever.k = 1

embedding = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
faiss_vectorstore = FAISS.from_texts(
    doc_list, embedding, metadatas=[{"source": i} for i in range(len(data))]
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k":1})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.2, 0.8]
)

def search(query):
    ensemble_docs = ensemble_retriever.invoke(query)
    return ensemble_docs

def chatgpt_generate(query):

    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    },{
        "role": "user",
        "content": query
    }]
    response = client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content
    return answer

def prompt_and_generate(query, docs, price):
    
    prompt = f"""아래 질문을 기반으로 검색된 뉴스를 참고하여 질문에 대한 답변을 생성하시오. 질문과 가장 적절한 뉴스에서 등장한 이벤트까지 출력하시오.
답변 마지막에 오늘의 종가 정보도 포함시켜서 답변을 만드시오.
종가: {price}

질문: {query}
"""
    
    for i in range(len(docs)):
        prompt += f"뉴스{i+1}\n"
        prompt += f"요약: {docs[i].page_content}\n"
        idx = docs[i].metadata['source']
        prompt += f"카테고리: {data[idx]['문서 카테고리']}\n"
        prompt += f"이벤트: {data[idx]['주요 이벤트']}\n"
        prompt += "\n"

    print(prompt)
    answer = chatgpt_generate(prompt)
    return answer

def first_chain(query):
    prompt = """아래 질문에 기업명이 포함되어 있다면, 기업번호(ticker)를 출력하시오. 반드시 ticker만을 출력하고, 기업명이 포함되어 있지 않은 경우는 0을 출력하시오.
출력 예시: {'기업명': '기업번호'}\n"""
    prompt += f"질문: {query}"
    answer = chatgpt_generate(prompt)
    return answer

query = "삼성전자가 인수하는 기업은?"
companies = first_chain(query)
company_dict = eval(companies)
tickers = list(company_dict.values())

today = datetime.today().strftime("%Y%m%d")
prices = stock.get_market_ohlcv(today, today, tickers[0])

if len(prices) > 0:
    price = prices.iloc[0, 3]
else:
    price = "모름"

retrieved = [doc for doc in search(query)]
answer = prompt_and_generate(query, retrieved, price)
print(answer)