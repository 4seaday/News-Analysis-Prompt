from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
import transformers
import torch

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

embedding = SentenceTransformerEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
faiss_vectorstore = FAISS.from_texts(
    doc_list, embedding, metadatas=[{"source": i} for i in range(len(data))]
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k":1})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

model_id = "42dot/42dot_LLM-SFT-1.3B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16}
)

pipeline.model.eval()

def search(query):
    ensemble_docs = ensemble_retriever.invoke(query)
    return ensemble_docs

def sllm_generate(query):

    answer = pipeline(
        query,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.5,
        top_p=0.9
    )
    return answer[0]['generated_text'][len(query):]

def prompt_and_generate(query, docs):
    
    prompt = f"""아래 질문을 기반으로 검색된 뉴스를 참고하여 질문에 대한 답변을 생성하시오.

질문: {query}
"""
    
    for i in range(len(docs)):
        prompt += f"뉴스{i+1}\n"
        prompt += f"요약: {docs[i].page_content}\n"
        prompt += "\n"
    prompt += "답변: "
    
    answer = sllm_generate(prompt)
    return answer

query = "삼성전자가 인수하는 기업은?"
retrieved = [doc for doc in search(query)]
answer = prompt_and_generate(query, retrieved)
print(answer)