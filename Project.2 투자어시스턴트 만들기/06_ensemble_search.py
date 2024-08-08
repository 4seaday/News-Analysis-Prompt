from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
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

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.2, 0.8]
)

ensemble_docs = ensemble_retriever.invoke(query)
print(ensemble_docs)