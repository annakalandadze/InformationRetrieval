from rank_bm25 import BM25Okapi

if __name__ == '__main__':

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
    corpus = [
        "Hello there good man!",
        "It is quite windy in London",
        "How is the weather today?"
    ]

    tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    query = "windy London"
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)