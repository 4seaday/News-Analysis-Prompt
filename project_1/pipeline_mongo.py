from temp import config
import openai
from gdeltdoc import GdeltDoc, Filters
from newspaper import Article
from pymongo import MongoClient
import datetime

open_ai_key = config.OPEN_AI_API_KEY
openai.api_key = open_ai_key
model = "gpt-3.5-turbo-0125"
gd = GdeltDoc()
client = MongoClient(host="localhost", port=27017)
db = client["project1"]
collection = db["NewsAnalysis1"]

def chatgpt_generate(query):

    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    },{   
        "role": "user",
        "content": query}]

    response = openai.ChatCompletion.create(model=model, messages=messages)
    answer = response['choices'][0]['message']['content']
    return answer

def get_url(keyword):
    f = Filters(
        start_date = "2024-03-20",
        end_date = "2024-03-25",
        num_records = 10,
        keyword = keyword,
        domain = "nytimes.com",
        country = "US",
    )
    
    articles = gd.article_search(f)
    return articles

def url_crawling(df):
    urls = df["url"]
    titles = df["title"]
    texts = []
    for url in urls:
        article = Article(url)
        article.download()
        article.parse()
        texts.append(article.text)
    return texts, titles

def analysis():
    prompt = '''아래 뉴스에서 S&P500에 상장된 기업명을 모두 추출하고, 기업에 해당하는 감성을 분석하시오.
    각 감성에 스코어링을 하시오. 각 스코어의 합은 1이 되어야 합니다. 소수점 첫번째까지만 생성하세요.
    출력포맷은 리스트이며, 세부 내용은 다음과 같습니다.
    반드시 출력포맷만을 생성하시오. 그 이외의 단어나 설명은 생성하지마시오.
    [{"organization": <기업명>, "positive": 0~1, "negative": 0~1, "neutral": 0~1}, ...]

    뉴스: '''

    orgs = ["microsoft", "apple", "tesla"]
    for org in orgs:
        df = get_url(org)
        dates = df['seendate']
        texts, titles= url_crawling(df)
        for idx, text in enumerate(texts):
            news_item = {}
            answer = chatgpt_generate(prompt + text)
            try:
                answer_list = eval(answer)
                news_item["text"] = text
                news_item["title"] = titles[idx]
                [item.update({"seendate": dates[idx]}) for item in answer_list]
                news_item["sentiments"] = answer_list
                news_item["date"] = datetime.datetime.now()
                insert_id = collection.insert_one(news_item)
                print(insert_id)
            except:
                continue
    return

analysis()