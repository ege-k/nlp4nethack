from NetHackCorpus.corpus_loader import load_corpus
from gensim import corpora, models, similarities


class GensimCorpus:
    def __init__(self, corpus):
        self.corpus = corpus
        stoplist = set('for a of the and to in'.split(' '))
        texts = [[word for word in document.lower().split() if word not in stoplist]
                 for document in corpus]
        from collections import defaultdict
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]

        self.dictionary = corpora.Dictionary(processed_corpus)
        bow_corpus = [self.dictionary.doc2bow(text) for text in processed_corpus]

        self.lsi = models.LsiModel(bow_corpus, id2word=self.dictionary, num_topics=256)

        self.index = similarities.Similarity("lsi", self.lsi[bow_corpus], num_features=300)

    def search(self, query):
        query_bow = self.dictionary.doc2bow(query.split())
        sims = self.index[self.lsi[query_bow]]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        return sims[:10]


def load_index(corpus):
    return GensimCorpus(corpus)


if __name__ == "__main__":
    corpus = load_corpus("NetHackCorpus/")
    index = load_index(corpus)
    res = index.search("You open the door.")
    print(res)
    for i in res:
        print(i[1], corpus[i[0]][:100])
