# semantic-search-with-ann
An example recipe search engine using a document -> embedding model (sentence transformer), an approximate nearest neighbors index, and a query / document re-ranker.

Used:
SBERT (sentence bert) for the language models and Spotify's Annoy package for ANN indexing.

It comes with a simple flask API and frontend for searching recipes (uses Bulma for CSS)

Docker and docker-compose was used in this repository, so should be easy to run.

## Quick Start
First download the RAW_recipes.csv dataset from Kaggle:
https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv

Place the RAW_recipes.csv file into the "data" folder, creating the folder if it doesn't exist.

Then build the python docker image
> docker-compose build

Run the following script, which will download sentence models from SBERT, embed the recipe documents, and build the ann index.
> docker-compose run semantic_search python build_index.py 

Finally, run the API with:
> docker-compose up

Then go to http://localhost:8080/ to the search page.

Note, that for ease of starting, the initial dataset is limited to just 1000 results, you can change the NROWS variable
in semantic_search/documents.py to increase that number. It will take longer to build the ANN index the larger that number is.


## Credit
Shout out to this example:
https://colab.research.google.com/github/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb#scrollTo=UlArb7kqN3Re

And this tutorial and explanation:
https://www.sbert.net/examples/applications/semantic-search/README.html#retrieve-re-rank

For the ML sentence transformer models:
SBERT
https://www.sbert.net/

And Hugging Face
https://huggingface.co/


Spotify, for the ANN index
https://github.com/spotify/annoy


And Bulma for the awesome CSS package:
https://bulma.io/

