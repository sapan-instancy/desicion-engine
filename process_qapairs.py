import logging, os, process_text as pt, re, pandas as pd, xlrd
from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

temp_directory = r'C:\Users\testuser\Documents\desicion_engine_1\tmp'
qa_file = r'C:\Users\testuser\Documents\desicion_engine_1\question-answers.xlsx'
df = pd.read_excel(qa_file)

def process_qapairs():
    print(df)


def create_vectors():
    documents = df['question'].tolist()
    # remove common words and tokenize
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1]
            for text in texts]
    dictionary = corpora.Dictionary(texts)
    dictionary.save(os.path.join(temp_directory, 'pre_questions.dict'))
    #convert tokenized docs to vectors
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(os.path.join(temp_directory, 'pre_questions.mm'), corpus)



def transform_to_tfidf(input_text):
    dictionary = None
    corpus = None
    if (os.path.exists(os.path.join(temp_directory, 'pre_questions.dict'))):
        dictionary = corpora.Dictionary.load(os.path.join(temp_directory, 'alltext.dict'))
    if (os.path.exists(os.path.join(temp_directory, 'pre_questions.mm'))):
        corpus = corpora.MmCorpus(os.path.join(temp_directory, 'pre_questions.mm'))

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    index = similarities.MatrixSimilarity(corpus_tfidf)
    index.save(os.path.join(temp_directory, 'pre_questions.index'))

    vec_bow = dictionary.doc2bow(input_text.lower().split())
    vec_lsi = tfidf[vec_bow]
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])  # sort in descending order

    relv_sims = sims[0:3]
    print(relv_sims)
    id, acc = relv_sims[0]
    return df.loc[id]

# create_vectors()
pprint(transform_to_tfidf('what are some common ergonomics types of injuries'))