import spacy  # version 3.5
from newspaper import Article

## Gelt API
def gdelt_api():
    from gdeltdoc import GdeltDoc, Filters

    f = Filters(
        keyword = "apple",
        start_date = "2024-02-24",
        end_date = "2024-03-25",
        country = "US",
        domain = "nytimes.com",
    )

    gd = GdeltDoc()

    # Search for articles matching the filters
    articles = gd.article_search(f)
    # print(len(article))

    # Get a timeline of the number of articles matching the filters
    timeline = gd.timeline_search("timelinevol", f)
    return articles

def parse_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def main():
    df = gdelt_api()
    url = df.iloc[0, 0]
    text = parse_text(url)

    nlp = spacy.load("en_core_web_md")

    # add pipeline (declared through entry_points in setup.py)
    nlp.add_pipe("entityLinker", last=True)

    doc = nlp(text)

    # returns all entities in the whole document
    all_linked_entities = doc._.linkedEntities
    # iterates over sentences and prints linked entities

    for sent in doc.sents:
        print(sent)
        sent._.linkedEntities.pretty_print()
    
    return

if __name__ == "__main__":
    main()