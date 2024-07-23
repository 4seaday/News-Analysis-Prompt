from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
import config
from openai import OpenAI

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

def prompt_and_generate(query, docs):
    
    prompt = f"""아래 질문을 기반으로 검색된 뉴스를 참고하여 질문에 대한 답변을 생성하시오. 질문과 가장 적절한 뉴스에서 등장한 이벤트까지 출력하시오. 

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

query = "삼성전자가 인수하는 기업은?"
retrieved = [doc for doc in search(query)]
answer = prompt_and_generate(query, retrieved)
print(answer)