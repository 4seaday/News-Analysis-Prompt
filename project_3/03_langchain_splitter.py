from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

loader = PyPDFLoader("/Users/jeongminsu/Dropbox/04_jobs/11_FastCampus/03_코드/01_10개프로젝트LLM서비스개발/project_3/bok_sample.pdf")
pages = loader.load_and_split()

text = pages[0].page_content

text_splitter = CharacterTextSplitter(
    separator=' .\n',
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

texts = text_splitter.split_text(text)

print(texts[0])
print(len(texts[0]))
print(len(texts))