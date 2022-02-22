# Import the documents csv
import numpy as np
import pandas as pd
import os

from annoy import AnnoyIndex
from documents import documents, NROWS
from sentence_models import embed, model_name

if __name__ == '__main__':
    embeddings_location = f'../data/{model_name}_sentence_embeddings.npy'

    if os.path.isfile(embeddings_location):
        sentence_embeddings = np.load(embeddings_location)
    else:
        # Use then sentence model to map documents to embeddings
        print('Creating document embeddings...')
        sentence_embeddings = embed(documents['text'][:NROWS], show_progress_bar=True)
        print('Done!')
        np.save(embeddings_location, sentence_embeddings)
        print(f'Saved document embeddings to {embeddings_location}')

    print(sentence_embeddings.shape[1])
    vector_length = sentence_embeddings.shape[1]
    print(f'Embedding vector length: {vector_length}')

    # Use an approximate nearest neighbors index, to index our documents by embedding
    # Here we used the spotify library Annoy because it is easy to use
    # But there are many other choices. See:
    # https://ann-benchmarks.com
    print('Creating approximate nearest neighbors index...')
    index = AnnoyIndex(vector_length, "dot")
    for i, embedding in enumerate(sentence_embeddings):
        index.add_item(i, embedding)

    index.build(10)
    
    # This can be loaded later when we want to host it in an API.
    index_file_name = f'../data/{model_name}_ann_index.ann'
    index.save(index_file_name)
    print(f'Saved index to file {index_file_name}')
