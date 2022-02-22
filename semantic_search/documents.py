import pandas as pd
import json

# Load the documents that we want to embed
# In this case it is a set of recipes from
# https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv
NROWS = 10000
documents_file = '../data/RAW_recipes.csv'

print(f'Loading documents {documents_file}...')
documents = pd.read_csv('../data/RAW_recipes.csv', nrows=NROWS)

def parse_python_array_text(text):
    return text.strip('[]').replace("'", "").replace("-", " ").split(', ')

# Can get clever with this, like concatenating the ingredients, steps, or description to the recipe name
def get_text(row):
    ingredients = ', '.join(row['ingredients'])
    return str(row['name']) + ' ' + str(row['description']) + ' with ingredients ' + ingredients

documents['ingredients'] = documents['ingredients'].apply(parse_python_array_text)
documents['tags'] = documents['tags'].apply(parse_python_array_text)
documents['text'] = documents.apply(get_text, 1)
print(documents['text'][0])

print(f'Loaded {len(documents)} documents, first few documents: {documents.head()["text"]}')

def load_documents():
    return documents