import os
import models
import article_scrape
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/', methods=['GET'])
def home():
     return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        url = request.form.get('url', '')
        try:
            article_text = article_scrape.get_article_text(url)
        except Exception:
            return render_template('index.html', bias=f'There was an error scraping the article')

        # Set false to use the bert model prediction
        bert_pred = models.predict_BERT(article_text)
        roberta_pred = models.predict_roBERTa(article_text)
        svm_pred = models.predict_SVM(article_text)
        distilbert_pred = models.predict_distilBERT(article_text)

        return render_template(
            'index.html',
            bert=f'BERT model prediction: {bert_pred}',
            roberta=f'RoBERTa model prediction: {roberta_pred}',
            distilbert=f'DistilBERT model prediction: {distilbert_pred}',
            svm=f'Linear model prediction: {svm_pred}'
        )

    return jsonify('Success...')
    

