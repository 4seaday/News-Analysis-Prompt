from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/Users/jeongminsu/Dropbox/04_jobs/11_FastCampus/03_코드/01_10개프로젝트LLM서비스개발/project_3/bok_sample.pdf")
pages = loader.load_and_split()

text = pages[0].page_content
print(text)