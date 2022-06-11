
import pandas as pd
import os
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import numpy as np
from gensim import corpora, models
import gensim
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

file_path = r'/Users/huchangguo/Desktop/Indiegogo'
data = pd.DataFrame()
df1 = pd.read_csv(os.path.join(file_path,'Indiegogo1.csv'),usecols=(['title']))
df2 = pd.read_csv(os.path.join(file_path,'Indiegogo2.csv'),usecols=(['title']))
df3 = pd.read_csv(os.path.join(file_path,'Indiegogo3.csv'),usecols=(['title']))
df4 = pd.read_csv(os.path.join(file_path,'Indiegogo4.csv'),usecols=(['title']))
df5 = pd.read_csv(os.path.join(file_path,'Indiegogo5.csv'),usecols=(['title']))
data = pd.concat([df1,df2,df3,df4,df5])

data

# Initialize regex tokenizer
tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = get_stop_words('en')
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
Dlist = data['title'].to_list()
Dlist = [str(x) for x in Dlist]
Dlist = [x for x in Dlist if x != 'nan']

len(Dlist)

T = []
for i in Dlist:
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    T.append(stemmed_tokens)    
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(T)
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in T]
T

for D in corpus:
    print([[dictionary[id], freq] for id, freq in D])
tfidf = models.TfidfModel(corpus, smartirs='ntc')
Tcorpus = tfidf[corpus]

for D in Tcorpus:
    print([[dictionary[id], np.around(freq,2)] for id, freq in D])
    
 #LSA 
# Vectorize document using TF-IDF
tfidf = TfidfVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = tokenizer.tokenize)
# Fit and Transform the documents
# train_data = tfidf.fit_transform(doc_list)
train_data = tfidf.fit_transform(data['title'].values.astype('U'))

# Define the number of topics or components
num_components=20
# Create SVD object
lsa = TruncatedSVD(n_components=num_components, n_iter=100,random_state=42)
# Fit SVD model on data
lsa.fit_transform(train_data)
# Get Singular values and Components
Sigma = lsa.singular_values_
V_transpose = lsa.components_.T
# Print the topics with their terms
terms = tfidf.get_feature_names()
for index, component in enumerate(lsa.components_):
    zipped = zip(terms, component)
    top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:5]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(index)+": ",top_terms_list)

  
 #LDA
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=20)
Dlda = ldamodel[corpus]
print(ldamodel.print_topics())

 #Visualize of LDA
pyLDAvis.enable_notebook()
V = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
V
    

import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode()
import plotly.graph_objs as go
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist, squareform
from gensim.matutils import jensen_shannon
from scipy import spatial as scs
    
    
 topic_dist = ldamodel.state.get_lambda()

num_words = 300
topic_terms = [{w for (w, _) in ldamodel.show_topic(topic, topn=num_words)} for topic in range(topic_dist.shape[0])]
n_ann_terms = 10

# use Euclidean distance metric in dendrogram
def e_dist(X):
   
    return pdist(X,'euclidean')


# define method for distance calculation in clusters
linkagefun=lambda x: sch.linkage(x, 'single')

def text_annotation(topic_dist, topic_terms, n_ann_terms, linkagefun):

    linkagefun = lambda x: sch.linkage(x, 'single')
    d = e_dist(topic_dist)
    Z = linkagefun(d)
    P = sch.dendrogram(Z, orientation="bottom", no_plot=True)

    x_ticks = np.arange(5, len(P['leaves']) * 10 + 5, 10)
    x_topic = dict(zip(P['leaves'], x_ticks))

    topic_vals = dict()
    for key, val in x_topic.items():
        topic_vals[val] = (topic_terms[key], topic_terms[key])

    text_annotations = []

    for trace in P['icoord']:
        fst_topic = topic_vals[trace[0]]
        scnd_topic = topic_vals[trace[2]]
        
        pos_tokens_t1 = list(fst_topic[0])[:min(len(fst_topic[0]), n_ann_terms)]
        neg_tokens_t1 = list(fst_topic[1])[:min(len(fst_topic[1]), n_ann_terms)]

        pos_tokens_t4 = list(scnd_topic[0])[:min(len(scnd_topic[0]), n_ann_terms)]
        neg_tokens_t4 = list(scnd_topic[1])[:min(len(scnd_topic[1]), n_ann_terms)]

        t1 = "<br>".join((": ".join(("+++", str(pos_tokens_t1))), ": ".join(("---", str(neg_tokens_t1)))))
        t2 = t3 = ()
        t4 = "<br>".join((": ".join(("+++", str(pos_tokens_t4))), ": ".join(("---", str(neg_tokens_t4)))))

        if trace[0] in x_ticks:
            t1 = str(list(topic_vals[trace[0]][0])[:n_ann_terms])
        if trace[2] in x_ticks:
            t4 = str(list(topic_vals[trace[2]][0])[:n_ann_terms])

        text_annotations.append([t1, t2, t3, t4])

        intersecting = fst_topic[0] & scnd_topic[0]
        different = fst_topic[0].symmetric_difference(scnd_topic[0])

        center = (trace[0] + trace[2]) / 2
        topic_vals[center] = (intersecting, different)

        topic_vals.pop(trace[0], None)
        topic_vals.pop(trace[2], None)  
        
    return text_annotations

   
 annotation = text_annotation(topic_dist, topic_terms, n_ann_terms, linkagefun)

#dendrogram
dendro = ff.create_dendrogram(topic_dist, labels=range(1, 21), linkagefun=linkagefun, hovertext=annotation)
dendro['layout'].update({'width': 1000, 'height': 600})
py.iplot(dendro)   
    
    
    
    
    
    
    
    
    
    
