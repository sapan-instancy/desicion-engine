import logging, os, process_text as pt, re, pandas as pd, xlrd
from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

temp_directory = r'C:\Users\testuser\Documents\desicion_engine_1\tmp'
qa_file = r'C:\Users\testuser\Documents\desicion_engine_1\question-answers.xlsx'
df = pd.read_excel(qa_file)


# This private function converts text based data to vectors reqd for applying different Information retrival techniques
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


# This method applies Normalized tfidf indexing over the corpus and then returns similar elements to the input text
def transform_to_tfidf_and_query(input_text):

    create_vectors()

    dictionary = None
    corpus = None
    if (os.path.exists(os.path.join(temp_directory, 'pre_questions.dict'))):
        dictionary = corpora.Dictionary.load(os.path.join(temp_directory, 'alltext.dict'))
    if (os.path.exists(os.path.join(temp_directory, 'pre_questions.mm'))):
        corpus = corpora.MmCorpus(os.path.join(temp_directory, 'pre_questions.mm'))

    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=5)
    corpus_lda = lda[corpus]
    index = similarities.MatrixSimilarity(corpus_lda)
    index.save(os.path.join(temp_directory, 'alltext_lda.index'))

    vec_bow = dictionary.doc2bow(input_text.lower().split())
    vec_lsi = lda[vec_bow]
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])  # sort in descending order

    relv_sims = sims[0:3]
    print(relv_sims)
    id, acc = relv_sims[0]
    return df.loc[id]


pprint(transform_to_tfidf('correct head position'))