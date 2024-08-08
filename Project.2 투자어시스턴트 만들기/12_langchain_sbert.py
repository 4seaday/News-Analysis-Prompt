from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

doc_list = [
    "우리나라는 2022년에 코로나가 유행했다.",
    "우리나라 2024년 GDP 전망은 3.0%이다.",
    "우리나라는 2022년 국내총생산 중 연구개발 예산은 약 5%이다."
]

embedding = SentenceTransformerEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
faiss_vectorstore = FAISS.from_texts(
    doc_list, embedding, metadatas=[{"source": 1}] * len(doc_list)
)
faiss_retriver = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

query = "2022년 우리나라 GDP대비 R&D 규모는?"

docs = faiss_retriver.invoke(query)
print(docs)

