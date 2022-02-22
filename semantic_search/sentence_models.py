# The lovely people at SBERT have provided free sentence transformer models
from sentence_transformers import CrossEncoder, SentenceTransformer
import os

def fine_tune_biencoder_model(be_model: SentenceTransformer):
    # Optionally train your model with your own data
    return be_model

# There are many to choose from, see:
# https://www.sbert.net/docs/pretrained_models.html
# I went with this one because it was trained on query -> document pairs
model_name = 'msmarco-distilbert-dot-v5'
model_dir_path = '../data/models/' + model_name

# Load model from file if file exists
if os.path.isdir(model_dir_path):
    model = SentenceTransformer(model_dir_path)
else:
    model = SentenceTransformer(model_name)
    fine_tune_biencoder_model(model)
    model.save(model_dir_path)

vector_size = model.get_sentence_embedding_dimension()

def embed(sentences, show_progress_bar=False):
    return model.encode(sentences, show_progress_bar=show_progress_bar)


# We use a cross-encoder, to re-rank the results after the bi-encoder / ann step.
# This dramatically improves the result quality
def fine_tune_crossencoder_model(ce_model: CrossEncoder):
    # Optionally train your model with your own data
    return ce_model


cross_encoder_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
cross_encoder_model_dir_path = '../data/models/' + cross_encoder_name
# Load model from file if file exists
if os.path.isdir(cross_encoder_model_dir_path):
    cross_encoder = CrossEncoder(cross_encoder_model_dir_path)
else:
    cross_encoder = CrossEncoder(cross_encoder_name)
    fine_tune_crossencoder_model(cross_encoder)
    cross_encoder.save(cross_encoder_model_dir_path)
