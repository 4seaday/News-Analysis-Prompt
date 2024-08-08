from platform import java_ver
from gdeltdoc import GdeltDoc, Filters
from newspaper import Article
import pandas as pd

gd = GdeltDoc()

def get_urls(keyword: str, start_date: str, end_date: str):
    f = Filters(
        start_date=start_date,
        end_date=end_date,
        num_records=10,
        keyword=keyword,
        domain="nytimes.com",
        country='US'
    )

    articles = gd.article_search(f)
    return articles

def parse_text(article_df: pd.DataFrame):

    result = []
    for _, row in article_df.iterrows():
        temp = {}
        url = row['url']
        article = Article(url)
        article.download()
        article.parse()
        title = row['title']
        date = row['seendate']
        temp['title'] = title
        temp['date'] = date
        temp['text'] = article.text
        result.append(temp)
    return result

total_list = []
keywords = ['inflation', 'real estate']
for keyword in keywords:
    df = get_urls(keyword, start_date="2024-03-25", end_date="2024-03-27")
    result_list = parse_text(df)
    total_list.extend(result_list)

for line in total_list:
    print(line)
    print()

unique_list = []
unique_titles = []
for line in total_list:
    if line['title'] not in unique_titles:
        unique_list.append(line)
    else:
        continue