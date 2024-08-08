from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import config

embedding_model = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)

loader = PyPDFLoader("/Users/jeongminsu/Dropbox/04_jobs/11_FastCampus/03_코드/01_10개프로젝트LLM서비스개발/project_3/bok_sample.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    separator=' .\n',
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

texts = text_splitter.split_documents(documents)
db = FAISS.from_documents(texts, embedding_model)

retriever = db.as_retriever()
docs = retriever.invoke("2022년 우리나라 GDP대비 R&D 규모는?")

print(docs)