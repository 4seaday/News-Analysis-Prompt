from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import config

doc_list = [
    "우리나라는 2022년에 코로나가 유행했다.",
    "우리나라 2024년 GDP 전망은 3.0%이다.",
    "우리나라는 2022년 국내총생산 중 연구개발 예산은 약 5%이다."
]

bm25_retriever = BM25Retriever.from_texts(
    doc_list, metadatas=[{"source": 1}]*len(doc_list)
)
bm25_retriever.k = 1

embedding = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
faiss_vectorstore = FAISS.from_texts(
    doc_list, embedding, metadatas=[{"source": 1}] * len(doc_list)
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k":1})

query = "2022년 우리나라 GDP대비 R&D 규모는?"

bm25_docs = bm25_retriever.invoke(query)
faiss_docs = faiss_retriever.invoke(query)

print(bm25_docs)
print('-----------')
print(faiss_docs)