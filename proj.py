import os, os.path
import sys
import time
from sklearn.feature_extraction.text import CountVectorizer
from math import log
from nltk import sent_tokenize
import operator

inv_index = {}
idfs = {}
n_docs = 0
term_count = {}

def idf(term):
    global n_docs 
    N = n_docs + 1 # total number of documents
    df = len(inv_index[term]) # total number of documents where term appears
    result = log(N/df)
    return result

def store_idfs():
    for term in inv_index.keys():
        idfs[term] = idf(term)

def create_tfidf_dict(sentences,d):
    tfidf_dict = {}
    for sentence in sentences:
        term_tfidf = 0
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([sentence])
        fileterms = vectorizer.get_feature_names_out()
        total_term_count = term_count[d]
        for term in fileterms:
            for doc in inv_index[term]:
                if doc[0] == d:
                    tn = doc[1]
            term_tfidf += idfs[term] * (tn / total_term_count)
        tfidf_dict[sentence] = term_tfidf / len(fileterms)
    return tfidf_dict

def index_document(filename):
    global n_docs
    file = open(filename,"r")
    filebody = file.read()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([filebody])
    fileterms = vectorizer.get_feature_names_out()
    term_count[filename] = len(fileterms)
    matrix = X.toarray()[0]
    for token in fileterms:
        if inv_index.get(token) == None:
            inv_index[token] = [(filename,matrix[vectorizer.vocabulary_[token]])]
        elif filename not in inv_index[token]:
            inv_index[token].append((filename,matrix[vectorizer.vocabulary_[token]]))
    n_docs += 1
    file.close()
            
def indexing(D,args):
    start = time.time()
    for dirName, subdirList, fileList in os.walk("." + os.sep + D):
        for name in fileList:
            index_document(os.path.join(dirName, name))
    end = time.time()
    time_passed = end-start
    print("Indexing time passed: %.2fs." % time_passed)
    print("Indexing memory space: " + str(sys.getsizeof(inv_index)) + " bytes.")

def ranking(d,order,I,p=7,l=1000000):
    file = open(d,"r")
    filebody = file.read()
    sentences = sent_tokenize(filebody)
    tfidf_dict = create_tfidf_dict(sentences,d)
    if order == "relevance":
        tfidf_dict = dict(sorted(tfidf_dict.items(), key=operator.itemgetter(1),reverse=True))
    summary = ""
    sentence_count = 1
    char_count = 0
    for entry in tfidf_dict.keys():
        char_count += len(entry)
        if sentence_count < p and char_count < l:
            summary += entry
        else:
            break
        sentence_count += 1
    file.close()
    return summary

docs = sys.argv[1]
doc = sys.argv[2]
indexing(docs,None)
store_idfs()
print(ranking(doc,"relevance",inv_index))
