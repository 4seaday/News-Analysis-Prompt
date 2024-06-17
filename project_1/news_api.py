from gdeltdoc import GdeltDoc, Filters
from newspaper import Article

f = Filters(
    start_date = "2024-03-20",
    end_date = "2024-03-25",
    num_records = 250,
    keyword = "microsoft",
    domain = "nytimes.com",
    country = "US",
)

gd = GdeltDoc()

# Search for articles matching the filters
articles = gd.article_search(f)
url = articles.loc[1, "url"]
print(articles.loc[1, "title"])
print('-------------------------')
article = Article(url)
article.download()
article.parse()
print(article.text)