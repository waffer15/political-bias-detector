from newsplease import NewsPlease

def get_article_text(url):
    article = NewsPlease.from_url(url, timeout=10)
    return article.maintext
                    