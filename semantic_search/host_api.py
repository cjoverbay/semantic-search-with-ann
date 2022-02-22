import pandas as pd

from annoy import AnnoyIndex
from documents import load_documents
from flask import Flask, request, render_template
from sentence_models import embed, model_name, vector_size, cross_encoder

MAX_ANN = 100

documents = load_documents()
document_index = AnnoyIndex(vector_size, 'dot')
document_index.load(f'../data/{model_name}_ann_index.ann')
embed(["hello world"])
print("Loaded index, model, and documents")

def log(message):
    print(message, flush=True)

def search(query, n=5):
    query_embedding = embed([query])
    log(query)
    nns = document_index.get_nns_by_vector(query_embedding[0], n, include_distances=True)
    return documents.iloc[nns[0]]

app = Flask(__name__)

# Flask web page
@app.route('/')
def index():
    query = request.args.get('query', 'bananas')
    n = int(request.args.get('n', '9'))
    
    # First, use the ann + bi-encoder SentenceTransformer model to get nearest neighbors
    # Get 10X the number we want for nearest neighbors
    ann_results = search(query, min(n*3, MAX_ANN))

    log(ann_results)

    # Then, re-rank using the cross encoder
    cross_inp = [[query, result['text']] for i, result in ann_results.iterrows()]

    log(f'Running cross encoder on {ann_results.shape[0]} documents')
    cross_scores = cross_encoder.predict(cross_inp, show_progress_bar=True)
    ann_results['cross_score'] = cross_scores
    ann_results = ann_results.sort_values('cross_score', ascending=False)

    # Return top n results
    return render_template('index.html', n=n, query=query, search_results=ann_results[:n])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

