import logging, os, process_text as pt, re
from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
temp_directory = r'C:\Users\testuser\Documents\desicion_engine_1\tmp'



ip_directory_path = r'\\HOMEDRIVE2\Instancy Shared Data\sapan\Content Page by Page\content\pages'
result_df = pt.process_text(ip_directory_path)


# This private function converts text based data to vectors reqd for applying different Information retrival techniques
def create_vectors():
    df = result_df
    documents = df[df['type'] == 'text']['value'].tolist()
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
    dictionary.save(os.path.join(temp_directory, 'alltext.dict'))
    # pprint(dictionary.token2id)
    #convert tokenized docs to vectors
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(os.path.join(temp_directory, 'alltext.mm'), corpus)
    #todo: code for corpus streaming




# This method applies Latent semantic indexing over the corpus and then returns similar elements to the input text
def transform_to_lsi_and_query(input_text):

    create_vectors()

    dictionary = None
    corpus = None
    if (os.path.exists(os.path.join(temp_directory, 'alltext.dict'))):
        dictionary = corpora.Dictionary.load(os.path.join(temp_directory, 'alltext.dict'))
    if (os.path.exists(os.path.join(temp_directory, 'alltext.dict'))):
        corpus = corpora.MmCorpus(os.path.join(temp_directory, 'alltext.mm'))

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary)
    corpus_lsi = lsi[corpus_tfidf]
    corpora.MmCorpus.serialize(os.path.join(temp_directory, 'tfidf_corpus.mm'), corpus_lsi)
    index = similarities.MatrixSimilarity(corpus_lsi)
    index.save(os.path.join(temp_directory, 'alltext.index'))

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary)
    vec_bow = dictionary.doc2bow('Learn about neck position'.lower().split())
    vec_lsi = lsi[vec_bow]
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])  # sort in descending order
    relv_sims = sims[0:2]
    print(relv_sims)
    return [return_df_row(id, relevance) for (id, relevance) in relv_sims if
            return_df_row(id, relevance)['text'] != None]



# This method applies Latent Dirichlet allocation method over the corpus and then returns similar elements to the input text
def transform_to_lda_and_query(input_text):
    dictionary = None
    corpus = None
    if (os.path.exists(os.path.join(temp_directory, 'alltext.dict'))):
        dictionary = corpora.Dictionary.load(os.path.join(temp_directory, 'alltext.dict'))
    if (os.path.exists(os.path.join(temp_directory, 'alltext.dict'))):
        corpus = corpora.MmCorpus(os.path.join(temp_directory, 'alltext.mm'))

    # tfidf = models.TfidfModel(corpus)
    # corpus_tfidf = tfidf[corpus]
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
    return [return_df_row(id, relevance) for (id, relevance) in relv_sims if
          return_df_row(id, relevance)['text'] != None ]





# This method applies Normalized tfidf indexing over the corpus and then returns similar elements to the input text
def transform_to_tfidf_and_query(input_text):
    dictionary = None
    corpus = None
    if (os.path.exists(os.path.join(temp_directory, 'alltext.dict'))):
        dictionary = corpora.Dictionary.load(os.path.join(temp_directory, 'alltext.dict'))
    if (os.path.exists(os.path.join(temp_directory, 'alltext.mm'))):
        corpus = corpora.MmCorpus(os.path.join(temp_directory, 'alltext.mm'))

    tfidf = models.TfidfModel(corpus, normalize=True)
    corpus_tfidf = tfidf[corpus]

    index = similarities.MatrixSimilarity(corpus_tfidf)
    index.save(os.path.join(temp_directory, 'alltext_tfidf.index'))

    vec_bow = dictionary.doc2bow(input_text.lower().split())
    vec_lsi = tfidf[vec_bow]
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])  # sort in descending order

    relv_sims = sims[0:3]
    print(sims)
    return [return_df_row(id, relevance) for (id, relevance) in relv_sims if
          return_df_row(id, relevance)['text'] != None ]



#this is a private method used to return the selected row based on the 'id' from a tuple element on the similarity list
def return_df_row(id, relevance):
   return { 'pageNum': id, 'pageId': result_df.loc[id]['pageId'], 'elementId': result_df.loc[id]['elementid'], 'text': result_df.loc[id]['value'] }



# trims text to 71 characters for display in cards
def make_text_snippet(value):
    if value != None:
        text = re.sub( '\n', '', value)
        return text[0:70] + '....'
    else: return value





## to test this modules methods, uncomment the lines below
# pprint(transform_to_tfidf_and_query('shoulders'))
# pprint(transform_to_lda('position shoulders correctly'))