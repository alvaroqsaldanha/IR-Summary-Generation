import os, os.path
import re
import sys
import time
from urllib.parse import _NetlocResultMixinStr
from sklearn.feature_extraction.text import CountVectorizer
from math import log
from nltk import sent_tokenize
import operator
from dominate import document
from dominate.tags import *
import matplotlib.pyplot as plt 
import itertools
import spacy
from scipy import spatial

inv_index = {}
idfs = {}
n_docs = 0
term_count = {}
avg_dl = 0

#################### TEXT PROCESSING AND INDEXING ####################

def add_noun_phrases(filebody,filename):
  nlp = spacy.load('en_core_web_sm')
  doc = nlp(filebody)
  count_noun_phrases = 0
  for noun_phrase in doc.noun_chunks:
    if inv_index.get(noun_phrase.text) == None:
      inv_index[noun_phrase.text] = [(filename,filebody.count(noun_phrase.text))]
    else:
      inv_index[noun_phrase.text].append((filename,filebody.count(noun_phrase.text)))
    count_noun_phrases+=1
  term_count[filename] += count_noun_phrases

def idf(term):
    global n_docs 
    N = n_docs + 1 # total number of documents
    df = len(inv_index[term]) # total number of documents where term appears
    result = log(N/df)
    return result

def store_idfs():
    for term in inv_index.keys():
        idfs[term] = idf(term)

def create_tf_dict(sentences,d):
    tf_dict = {}
    total_term_count = term_count[d]
    for sentence in sentences:
        term_tf = 0
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([sentence])
        fileterms = vectorizer.get_feature_names_out()
        for term in fileterms:
            for doc in inv_index[term]:
                if doc[0] == d:
                    tn = doc[1]
            term_tf += tn / total_term_count
        tf_dict[sentence] = term_tf / len(fileterms)
    return tf_dict

def create_tfidf_dict(sentences,d):
    tfidf_dict = {}
    total_term_count = term_count[d]
    for sentence in sentences:
        term_tfidf = 0
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([sentence])
        fileterms = vectorizer.get_feature_names_out()
        for term in fileterms:
            for doc in inv_index[term]:
                if doc[0] == d:
                    tn = doc[1]
            term_tfidf += idfs[term] * (tn / total_term_count)
        tfidf_dict[sentence] = term_tfidf / len(fileterms)
    return tfidf_dict

def create_bm25_dict(sentences,d):
    bm25_dict = {}
    total_term_count = term_count[d]
    for sentence in sentences:
        term_bm25 = 0
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([sentence])
        fileterms = vectorizer.get_feature_names_out()
        for term in fileterms:
            for doc in inv_index[term]:
                if doc[0] == d:
                    tn = doc[1]
            term_bm25 += (idfs[term] * (2.2 * tn)) / (tn + 1.2 * (0.25 + 0.75 * (total_term_count / avg_dl)))
        bm25_dict[sentence] = term_bm25 / len(fileterms)
    return bm25_dict

def index_document(filename,preprocessing):
    global n_docs, avg_dl
    file = open(filename,"r")
    filebody = file.read()
    if preprocessing == "bigram":
        vectorizer = CountVectorizer(analyzer = "word", ngram_range=(1,2))
    else:
        vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([filebody])
    fileterms = vectorizer.get_feature_names_out()
    avg_dl += len(fileterms)
    term_count[filename] = len(fileterms)
    matrix = X.toarray()[0]
    for token in fileterms:
        if inv_index.get(token) == None:
            inv_index[token] = [(filename,matrix[vectorizer.vocabulary_[token]])]
        elif filename not in inv_index[token]:
            inv_index[token].append((filename,matrix[vectorizer.vocabulary_[token]]))
    n_docs += 1
    if preprocessing == "noun_phrases":
        add_noun_phrases(filebody,filename)
    file.close()
            
def indexing(D,preprocessing):
    global n_docs, avg_dl
    start = time.time()
    for dirName, subdirList, fileList in os.walk(D):
        for name in fileList:
            index_document(os.path.join(dirName, name),preprocessing)
    end = time.time()
    avg_dl = avg_dl / n_docs
    time_passed = end-start
    print("Indexing time passed: %.2fs." % time_passed)
    print("Indexing memory space: " + str(sys.getsizeof(inv_index)) + " bytes.")

def build_summary(sentence_dict,l):
    summary = ""
    char_count = 0
    for entry in sentence_dict.keys():
        char_count += len(entry)
        if char_count < l:
            summary += entry
        else:
            break
    return summary

def ranking(d,order,I,p=8,l=1000000,model="tfidf"):
    file = open(d,"r")
    filebody = file.read()
    filebody = re.sub(r"[!][\n]+",'!',filebody)
    filebody = re.sub(r"[?][\n]+",'?',filebody)
    filebody = re.sub(r"[:][\n]+",':',filebody)
    filebody = re.sub(r"[.][\n]+",'.',filebody)
    filebody = re.sub(r"[\n]+",'.',filebody)
    sentences = sent_tokenize(filebody)
    if model == "tdidf":
        sentence_dict = create_tfidf_dict(sentences,d)
    elif model == "tf":
        sentence_dict = create_tf_dict(sentences,d)
    else:
        sentence_dict = create_bm25_dict(sentences,d)
    sorted_dict = dict(sorted(sentence_dict.items(), key=operator.itemgetter(1),reverse=True))
    full_dict = sorted_dict
    min_relevance = 0
    if p <= len(sentence_dict.keys()):
        min_relevance =  sentence_dict[list(sorted_dict)[p-1]]
    if order == "relevance":
        sentence_dict = dict(itertools.islice(sorted_dict.items(), p))
    else:
        temp_dict = {}
        for sentence in sentence_dict.keys():
            if sentence_dict[sentence] >= min_relevance:
                temp_dict[sentence] = sentence_dict[sentence]
        sentence_dict = temp_dict
    summary = build_summary(sentence_dict,l)
    file.close()
    return (summary,sentence_dict, full_dict)

#################### RRF ####################

def rrf(rank_list):
  aggregated_rank = {}
  for sentence in rank_list[0].keys():
    rank = 1/(0.5 + rank_list[0][sentence])
    for model in rank_list[1:]:
      rank += 1/(0.5 + model[sentence])
    aggregated_rank[sentence] = rank
  sorted_rank = dict(sorted(aggregated_rank.items(), key=operator.itemgetter(1),reverse=True))
  return sorted_rank 

#################### MMR ####################

def create_mmr_rank(original_sentences,current_sentences,lb,summary):
  mmr_rank = {}
  original_doc = ''.join(original_sentences)
  current_doc = ''.join(current_sentences)
  for sentence in current_sentences:
    vectors = [original_doc,sentence,current_doc] 
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(vectors)
    sentence_vector = X.toarray()[1]
    curr_vector = X.toarray()[2]
    doc_similarity = spatial.distance.cosine(curr_vector,sentence_vector)
    aux = 0
    rank = (1 - lb) * doc_similarity 
    for summary_sentence in summary:
      vectors = [original_doc,summary_sentence] 
      vectorizer = CountVectorizer()
      X = vectorizer.fit_transform(vectors)
      summary_sentence_vector = X.toarray()[1]
      sentence_similarity = spatial.distance.cosine(sentence_vector,summary_sentence_vector)
      aux += sentence_similarity
    rank -= lb * aux
    mmr_rank[sentence] = rank
    sorted_mmr_rank = dict(sorted(mmr_rank.items(), key=operator.itemgetter(1),reverse=True)) 
    return sorted_mmr_rank

def create_mmr_summary(original_sentences,current_sentences,lb,mmr_rank,summary,p,l):
  most_relevant_sentence = list(mmr_rank.keys())[0]
  if p > 0 and (len(most_relevant_sentence) <= l):
    summary.append(most_relevant_sentence)
    p -= 1
    l -= len(most_relevant_sentence)
    current_sentences.remove(most_relevant_sentence)
    new_mmr_rank = create_mmr_rank(original_sentences,current_sentences,lb,summary)
    return create_mmr_summary(original_sentences,current_sentences,lb,new_mmr_rank,summary,p,l) 
  return summary 

def get_mmr_summary(doc,lb):
  file = open(doc,"r")
  filebody = file.read()
  filebody = re.sub(r"[!][\n]+",'!',filebody)
  filebody = re.sub(r"[?][\n]+",'?',filebody)
  filebody = re.sub(r"[:][\n]+",':',filebody)
  filebody = re.sub(r"[.][\n]+",'.',filebody)
  filebody = re.sub(r"[\n]+",'.',filebody)
  sentences = sent_tokenize(filebody)
  original_sentences = sentences.copy()
  mmr_rank = create_mmr_rank(original_sentences,sentences,lb,[])
  return create_mmr_summary(original_sentences,sentences,lb,mmr_rank,[],8,500)

#################### VISUALIZE ####################

def visualize(d,order,I,p=7,l=1000000,model="tfidf"):
    summary,sentence_dict,full_dict= ranking(d,order,I,p,l,model)
    file = open(d,"r")
    filebody = file.read()
    filebody = re.sub(r"[!][\n]+",'! ',filebody)
    filebody = re.sub(r"[?][\n]+",'? ',filebody)
    filebody = re.sub(r"[:][\n]+",': ',filebody)
    filebody = re.sub(r"[.][\n]+",'. ',filebody)
    filebody = re.sub(r"[\n]+",'. ',filebody)
    sentences = sent_tokenize(filebody)
    file.close()
    if order == "relevance":
        dict_list = list(sentence_dict)
    color = 70
    with document(title='Summary of ' + d) as doc:
        h1('Summary of ' + d )
        for sentence in sentences:
            if sentence_dict.get(sentence) != None:
                if order == "relevance":
                    curr_color = color + 30 * dict_list.index(sentence)
                    b(div(sentence,style="color: rgb(0," + str(curr_color) + ',0)'))
                else:
                    b(sentence)
            else:
                div(sentence)
        br()
        div("Sentence relevance in ascending order from lighter to darker colors.",style="font-size: 10px")
    f = open('summary_' + d + '.html', 'w')
    f.write(doc.render())
    f.close()
    return summary

#################### EVALUATION ####################

def create_results_directory(): 
    current_working_directory = os.path.abspath(os.getcwd())
    results_directory = "results"
    path = os.path.join(current_working_directory, results_directory)
    if not os.path.isdir(path): 
        os.mkdir(path)
        print("New results directory created")
    return path

def build_precision_recall_graph(doc,recall, precision, p, l): 
    font1 = {'family':'serif','color':'blue','size':20}
    font2 = {'family':'serif','color':'darkred','size':15}
    if p != None: 
        plt.title("precision-recall for " + str(p) + " sentence(s)",font1)
    else: 
        plt.title("precision-recall for " + str(l) + " characters",font1)
    plt.xlabel("recall", font2)
    plt.ylabel("precision", font2)
    plt.plot(recall,precision)
    path = create_results_directory()

    if not os.path.isdir(path+"\\"+doc):
        os.mkdir(path+"\\"+doc)
    filename = "precision_recall_graph_" 
    if p != None: 
        filename+= str(p) + "_sentences"
    else: 
        filename+= str(l) +"_char_length"
    plt.savefig(path+"\\"+doc+"\\"+ filename)


def build_precision_recall_curve(doc,ref_summary_sentences, summary_sentences, p, l): 
    total_retrieved_docs = 0
    relevant_retrieved_docs = 0
    recall = []
    precision = []
    for sentence in summary_sentences:
        total_retrieved_docs +=1
        if sentence in ref_summary_sentences: 
            relevant_retrieved_docs +=1 
            current_recall = relevant_retrieved_docs / len(ref_summary_sentences)
            recall.append(current_recall)
            current_precision = relevant_retrieved_docs / total_retrieved_docs
            precision.append(current_precision)
    build_precision_recall_graph(doc,recall, precision, p, l)

def mean_average_precision(rank, ref_summary_sentences):
    average_precision = 0
    relevant_docs = 0
    total_retrieved_docs = 0
    for sentence in rank.keys():
        total_retrieved_docs += 1
        if sentence in ref_summary_sentences:
            relevant_docs += 1
        average_precision += relevant_docs / total_retrieved_docs
    mean_average_precision = average_precision / len(ref_summary_sentences)
    print(mean_average_precision)

def summary_size_evaluation(doc,rank,ref_summary_sentences,P,L):
    for p_value in P:
        sentence_dict = dict(itertools.islice(rank.items(), p_value))
        summary = build_summary(sentence_dict,1000000)
        summary = re.sub(r'([a-z])\.([A-Z])', r'\1. \2',summary)
        summary = re.sub(r'([0-9])\.([A-Z])', r'\1. \2',summary)
        summary = re.sub(r'([a-z])\.([0-9])', r'\1. \2',summary)
        summary_sentences = sent_tokenize(summary)
        build_precision_recall_curve(doc,ref_summary_sentences,summary_sentences,p_value,None)
        mean_average_precision(sentence_dict,ref_summary_sentences)
    for l_value in L:
        summary = build_summary(rank,l_value)
        summary = re.sub(r'([a-z])\.([A-Z])', r'\1. \2',summary)
        summary = re.sub(r'([0-9])\.([A-Z])', r'\1. \2',summary)
        summary = re.sub(r'([a-z])\.([0-9])', r'\1. \2',summary)
        summary_sentences = sent_tokenize(summary)
        build_precision_recall_curve(doc,ref_summary_sentences,summary_sentences,None,l_value)
        mean_average_precision(sentence_dict,ref_summary_sentences)

def evaluation(D,S,I,P=[6,8,10,12],L=[100,200,300,400,500,750,1000,1500]):
    for doc,summaryfile in zip(D,S):
        full_rank = ranking(doc,"relevance",inv_index,p=8)
        rank = full_rank[2]
        file1 = open(summaryfile,"r")
        summarybody = re.sub(r'([a-z])\.([A-Z])', r'\1. \2',file1.read())
        summarybody = re.sub(r'([0-9])\.([A-Z])', r'\1. \2',summarybody)
        summarybody = re.sub(r'([a-z])\.([0-9])', r'\1. \2',summarybody)
        ref_summary_sentences = sent_tokenize(summarybody)
        summary_size_evaluation(doc.split("\\")[-1],rank,ref_summary_sentences,P,L)

### AUXILIARY FUNCTIONS ###

def build_document_list(path):
    evaluation_list = [] 
    for dirName, subdirList, fileList in os.walk(path):
            for name in fileList:
                evaluation_list.append(os.path.join(dirName,name))
    return evaluation_list

print("Please make sure BBC News Summary is in the same directory as this script.")
print("Directory structure should be .\BBC News Summary\\News Articles\\")
docs = '.\BBC News Summary\\News Articles\\'
ipt = ''

indexing(docs,None)
store_idfs()

while ipt != q:
    print("\nType and enter:")
    print("1 - to get summaries for a specific file")
    print("2 - to perform system evaluation")
    print("q - to exit")
    ipt = input('>>> ')
    if ipt == '1':
        file = input("Please enter the name of the file, including path:\n>>> ")
        summary = visualize(file,"relevance",inv_index)
        print("\nSummary for " + file + ":")
        print('\n' + summary)
        print('\nHTML of summary created in directory.')
    elif ipt == '2':
        testing_set = input("Introduce the path to the set of news you want to evaluate system performance:\n>>> ")
        reference_set = input("Introduce the corresponding summary set for comparison:\n>>> ")
        evaluation_list = build_document_list(testing_set)
        reference_list = build_document_list(reference_set)
        evaluation(evaluation_list, reference_list, inv_index)
    elif ipt == 'q':
        exit()
