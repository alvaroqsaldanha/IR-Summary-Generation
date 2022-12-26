from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import itertools
import operator

class graph:

    def __init__(self,nodes,edges):
        self.nodes = nodes
        self.edges = edges

    def __str__(self):
        repr = ""
        for i in range(len(self.nodes)):
            repr = repr + str(i) + " - " + self.nodes[i] + ": "
            for edge in self.edges[i]:
                repr += str(edge[0]) + "," + str(edge[1]) + "   "
            repr += '\n'    
        return repr
      
def build_graph(d,similarity_criteria,similarity_threshold):
    document_sentences = sent_tokenize(d)
    nodes = []
    edges = {}
    for i in range(len(document_sentences)):
        for j in range(i+1,len(document_sentences)):
            if similarity_criteria == "cosine_tfidf":
                tfidf_sentence_term_i = TfidfVectorizer()
                term_sentence_vector_i = tfidf_sentence_term_i.fit_transform([document_sentences[i]])
                term_sentence_dict_i = dict(zip(tfidf_sentence_term_i.get_feature_names_out(),term_sentence_vector_i.data))
                tfidf_sentence_term_j = TfidfVectorizer()
                term_sentence_vector_j = tfidf_sentence_term_j.fit_transform([document_sentences[j]])
                term_sentence_dict_j = dict(zip(tfidf_sentence_term_j.get_feature_names_out(),term_sentence_vector_j.data))
            similarity = cosine_similarity_graph(term_sentence_dict_i,term_sentence_dict_j)
            if similarity >= similarity_threshold:
                if edges.get(i) != None:
                    edges.get(i).append((j,similarity))
                else:
                    edges[i] = [(j,similarity)]
                if edges.get(j) != None:
                    edges.get(j).append((i,similarity))
                else:
                    edges[j] = [(i,similarity)]
        nodes.append(document_sentences[i])
    return graph(nodes,edges)

def cosine_similarity_graph(term_sentence_dict, term_document_dict):
    dot_product = 0
    sentence_mod = 0
    document_mod = 0
    for term in term_sentence_dict.keys():
        if term in term_document_dict.keys(): 
            dot_product += term_sentence_dict[term] * term_document_dict[term]
        sentence_mod += term_sentence_dict[term] * term_sentence_dict[term]
    for term in term_document_dict.keys():
        document_mod += term_document_dict[term] * term_document_dict[term]
    if dot_product == 0:
        return -1
    sentence_mod = np.sqrt(sentence_mod)
    document_mod = np.sqrt(document_mod)
    cosine_similarity = dot_product / (sentence_mod * document_mod)
    return cosine_similarity

def undirected_page_rank(d,p,similarity_criteria,similarity_threshold):
    graph = build_graph(d,"cosine_tfidf",similarity_threshold)
    E = np.zeros((len(graph.nodes),len(graph.nodes)))
    for i in range(len(graph.nodes)):
        if graph.edges.get(i) != None:
            for edge in graph.edges[i]:
                E[i,edge[0]] = 1
                E[edge[0],i] = 1
    for i in range(len(graph.nodes)):
        if E[:,i].sum() == 0:
            E[:,i] = 1
            E[i,:] = 1
            E[i,i] = 0
    M = np.zeros((len(graph.nodes),len(graph.nodes)))
    for column in range(len(graph.nodes)):
        out_links = E[:, column].astype('bool')
        total_n_out_links = out_links.sum()
        M[out_links, column] = 1/total_n_out_links
    damp_factor = 0.1
    n_iter = 20
    M = (M * (1-damp_factor)) + (damp_factor * 1/len(graph.nodes))
    r = np.full((len(graph.nodes)), 1/len(graph.nodes))
    for i in range(n_iter):
        r = np.dot(M, r)      
    sentence_dict = {graph.nodes[i]:r[i] for i in range(len(graph.nodes))}   
    sentence_dict = dict(sorted(sentence_dict.items(), key=operator.itemgetter(1),reverse=True))
    sentence_dict = dict(itertools.islice(sentence_dict.items(), p))
    return sentence_dict



        
