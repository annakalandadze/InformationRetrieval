from rank_bm25 import BM25Okapi
import csv
from sklearn.feature_extraction.text import CountVectorizer
import re

# Path to your queries.train.tsv file
queries_path = 'queries.train.tsv'

query_ids = []
query_texts = []
query_id_text = []

with open(queries_path, mode='r', encoding='utf-8', newline='') as tsv_file:
    tsv_reader = csv.DictReader(tsv_file, delimiter='\t')
    query_ids.append('121352')
    query_texts.append('define extreme')
    for row in tsv_reader:
        query_id = row['121352']
        query_text = row['define extreme']
        query_ids.append(query_id)
        query_texts.append(query_text)
        query_id_text.append((query_id, query_text))
    print(len(query_ids))

all_texts = query_texts
count_vectorizer = CountVectorizer()
corpus_term_freq = count_vectorizer.fit_transform(query_texts)
feature_names = count_vectorizer.get_feature_names_out()

query_term_freq_scores = {}
long_tail_queries = []
for (id, query) in query_id_text:
    number_of_rare = 0
    for word in re.split(r'[^a-zA-Z0-9]+', query):
        if word.lower() in count_vectorizer.vocabulary_:
            term_freq = count_vectorizer.vocabulary_[word.lower()]
            if term_freq < 8000:
                number_of_rare += 1
        else:
            number_of_rare += 1
    if number_of_rare > len(re.split(r'[^a-zA-Z0-9]+', query))/2.5 or len(re.split(r'[^a-zA-Z0-9]+', query)) > 40:
        long_tail_queries.append(id)
print(len(long_tail_queries))
with open('long_tail_queries.txt', 'w') as file:
    for id in long_tail_queries:
        file.write(str(id) + '\n')

# print(long_tail_queries)