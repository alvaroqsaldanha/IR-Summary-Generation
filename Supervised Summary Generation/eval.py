import matplotlib.pyplot as plt 
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import os  
from tqdm import tqdm
import numpy as np
import statistics
import operator

def build_summary(sentences,l):
    summary = ""
    char_count = 0
    sentence_count = 0
    for entry in sentences:
        char_count += len(entry)
        if char_count < l:
            summary += entry
            sentence_count += 1
        else:
            break
    return summary,sentence_count

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
    if(len(recall) == 0 ): 
        return 
    path = create_results_directory()
    if not os.path.isdir(path+"\\"+str(doc)):
        os.mkdir(path+"\\"+str(doc))
    filename = "precision_recall_graph_" 
    if p != None: 
        filename+= str(p) + "_sentences"
    else: 
        filename+= str(l) +"_char_length"
    plt.savefig(path+"\\"+ str(doc) + "\\"+ filename)
    plt.clf()

def build_f_measure_graph(mean_f_measure,X,x_axis_name,doc):
    if len(mean_f_measure) == 0: 
        return
    plt.title("plot f_measure in relation to " + x_axis_name) 
    plt.xlabel(x_axis_name)
    plt.ylabel("f_measure")
    plt.plot(X,mean_f_measure)
    path = create_results_directory()
    if not os.path.isdir(path+"\\"+str(doc)):
        os.mkdir(path+"\\"+str(doc))
    filename = "fmeasure_"+ x_axis_name
    plt.savefig(path+"\\"+ str(doc) + "\\"+ filename)
    plt.clf()

def build_precision_recall_curve(doc,ref_summary_sentences, summary_sentences, p, l): 
    total_retrieved_docs = 0
    relevant_retrieved_docs = 0
    recall = []
    precision = []
    f_measure = []
    for sentence in summary_sentences:
        total_retrieved_docs +=1
        if sentence in ref_summary_sentences: 
            relevant_retrieved_docs +=1 
            current_recall = relevant_retrieved_docs / len(ref_summary_sentences)
            recall.append(current_recall)
            current_precision = relevant_retrieved_docs / total_retrieved_docs
            precision.append(current_precision)
            f_measure.append(1.25*((current_precision*current_recall)/(0.25 * current_precision+current_recall)))
    if len(f_measure) == 0:
        mean_fmeasure = 0
    else:
        mean_fmeasure = sum(f_measure) / len(f_measure)
    build_precision_recall_graph(doc,recall, precision, p, l)
    return mean_fmeasure

def mean_average_precision(rank, ref_summary_sentences):
    average_precision = 0
    relevant_docs = 0
    total_retrieved_docs = 0
    for sentence in rank:
        total_retrieved_docs += 1
        if sentence in ref_summary_sentences:  
            relevant_docs += 1
            average_precision += relevant_docs / total_retrieved_docs
    if relevant_docs > 0: 
        mean_average_precision = average_precision  / relevant_docs
    else: 
        mean_average_precision = 0
    return mean_average_precision

def plot_map_variation(map, x_axis_name, x_axis_values, doc): 
    if len(map) == 0: 
        return
    plt.title("plot map variation in relation to " + x_axis_name) 
    plt.xlabel(x_axis_name)
    plt.ylabel("map")
    plt.plot(x_axis_values,map)
    path = create_results_directory()
    if not os.path.isdir(path+"\\"+str(doc)):
        os.mkdir(path+"\\"+str(doc))
    filename = "map_variation_in_"+ x_axis_name
    plt.savefig(path+"\\"+ str(doc) + "\\" + filename)
    plt.clf()

def summary_size_evaluation(produced_summary_sentences,ref_summary_sentences,P,L,doc):
    map_for_p = []
    mean_f_measure_for_p = []
    for p_value in P:
        summary_sentences = produced_summary_sentences[:p_value]
        mean_f_measure = build_precision_recall_curve(doc,ref_summary_sentences,summary_sentences,p_value,None)
        map = mean_average_precision(summary_sentences,ref_summary_sentences)
        map_for_p.append(map)
        mean_f_measure_for_p.append(mean_f_measure)
    plot_map_variation(map_for_p, "p_values", P, doc)
    build_f_measure_graph(mean_f_measure_for_p,P,"p",doc)
    map_for_l = []
    mean_f_measure_for_l = []
    for l_value in L:
        summary = build_summary(produced_summary_sentences,l_value)[0]
        summary = re.sub(r'([a-z])\.([a-z])', r'\1. \2',summary)
        summary = re.sub(r'([0-9])\.([a-z])', r'\1. \2',summary)
        summary = re.sub(r'([a-z])\.([0-9])', r'\1. \2',summary)
        summary_sentences = sent_tokenize(summary)
        mean_f_measure = build_precision_recall_curve(doc,ref_summary_sentences,summary_sentences,None,l_value)
        map = mean_average_precision(summary_sentences,ref_summary_sentences)
        map_for_l.append(map)
        mean_f_measure_for_l.append(mean_f_measure)
    plot_map_variation(map_for_l,"l_values", L, doc)
    build_f_measure_graph(mean_f_measure_for_l,L,"l",doc)
    return map_for_p, map_for_l

def get_full_summary(doc,M):
    article_sentences = sent_tokenize(doc) 
    produced_summary = {}
    count = 0
    for sentence in article_sentences:
        count += 1
        c = classify(sentence,doc,M,count,len(article_sentences))[1]
        produced_summary[sentence] = c
    sentence_dict = dict(sorted(produced_summary.items(), key=operator.itemgetter(1),reverse=True))
    return list(sentence_dict.keys())


def get_full_summary_rf(doc,summary,M):
    a = 0.75
    b = 1
    c = 0.01
    produced_summary = {}
    article_sentences = sent_tokenize(doc)
    ref_summary_senteces = sent_tokenize(summary) 
    n_relevant = len(ref_summary_senteces)
    n_irrelevant = len(article_sentences) - n_relevant
    features = {article_sentences[i]:feature_extraction(article_sentences[i],doc,i+1,len(article_sentences)) for i in range(len(article_sentences))}
    irrelevant_sentences = list(filter(lambda x: x not in ref_summary_senteces,article_sentences))
    relevant_sentences = list(filter(lambda x: x in ref_summary_senteces,article_sentences)) # Recalculating to avoid discrepancies in senteces of the reference summaries/articles 
    relevant_features_sum = [0,0,0,0]
    irrelevant_features_sum = [0,0,0,0]
    if len(relevant_sentences) > 0:
        relevant_features_sum = [sum(i) for i in zip(*[features[item] for item in relevant_sentences])]
    if len(irrelevant_sentences) > 0:
        irrelevant_features_sum = [sum(i) for i in zip(*[features[item] for item in irrelevant_sentences])]
    relevant_query = np.array([b * (1/n_relevant) * y for y in relevant_features_sum])
    irrelevant_query = np.array([c * (1/n_irrelevant) * x for x in irrelevant_features_sum])
    for sentence in article_sentences:
        original_query = np.array([a * z for z in features[sentence]])
        modified_query = np.add(original_query,np.subtract(relevant_query,irrelevant_query))
        modified_query = modified_query.tolist()
        c = classify_no_extraction(modified_query,M)[1]
        produced_summary[sentence] = c
    sentence_dict = dict(sorted(produced_summary.items(), key=operator.itemgetter(1),reverse=True))
    return list(sentence_dict.keys())  

def classify_no_extraction(feature,M):
    prob = M.predict_proba([feature])
    return [prob[:,1],prob[:,1]] 

def cosine_similarity(term_sentence_dict, term_document_dict, sentence_feature_names):
    dot_product = 0
    sentence_mod = 0
    document_mod = 0
    for term in sentence_feature_names: 
        dot_product += term_sentence_dict[term] * term_document_dict[term]
        sentence_mod += term_sentence_dict[term] * term_sentence_dict[term]
    for term in term_document_dict.keys():
        document_mod += term_document_dict[term] * term_document_dict[term]
    sentence_mod = np.sqrt(sentence_mod)
    document_mod = np.sqrt(document_mod)
    cosine_similarity = dot_product / (sentence_mod * document_mod)
    return cosine_similarity

def feature_extraction(s,d, sentence_count,num_sentences_in_article): 
    tfidf_sentence_term = TfidfVectorizer()
    term_sentence_vector = tfidf_sentence_term.fit_transform([s])
    term_sentence_dict = dict(zip(tfidf_sentence_term.get_feature_names_out(),term_sentence_vector.data))

    tfidf_document_term = TfidfVectorizer()
    term_document_vector = tfidf_document_term.fit_transform([d])
    term_document_dict = dict(zip(tfidf_document_term.get_feature_names_out(),term_document_vector.data))
    
    cos_similarity_tfidf = cosine_similarity(term_sentence_dict=term_sentence_dict, term_document_dict=term_document_dict, sentence_feature_names=tfidf_sentence_term.get_feature_names_out())

    tf_sentence_term = CountVectorizer()
    term_sentence_vector = tf_sentence_term.fit_transform([s])
    term_sentence_dict = dict(zip(tf_sentence_term.get_feature_names_out(),term_sentence_vector.data))

    tf_document_term = CountVectorizer()
    term_document_vector = tf_document_term.fit_transform([d])
    term_document_dict = dict(zip(tf_document_term.get_feature_names_out(),term_document_vector.data))

    cos_similarity_tf = cosine_similarity(term_sentence_dict=term_sentence_dict, term_document_dict=term_document_dict,sentence_feature_names=tf_sentence_term.get_feature_names_out())
    
    bm25_document_term_tf = CountVectorizer()
    term_document_vector_tf = bm25_document_term_tf.fit_transform([d])
    bm25_counts = term_document_vector_tf.data
    bm25_document_term_idf = TfidfVectorizer()
    term_document_vector = bm25_document_term_idf.fit_transform([d])
    bm25_idfs = term_document_vector.data
    n_features = len(bm25_document_term_idf.get_feature_names_out())
    avg_dl = n_features / num_sentences_in_article
    bm25_doc = [(bm25_idfs[i] * (2.2 * bm25_counts[i])) / (bm25_counts[i] + 1.2 * (0.25 + 0.75 * (n_features) / avg_dl)) for i in range(n_features)]
    term_document_dict = dict(zip(bm25_document_term_idf.get_feature_names_out(),bm25_doc))

    bm25_sentence_term_tf = CountVectorizer()
    term_sentence_vector_tf = bm25_sentence_term_tf.fit_transform([s])
    bm25_counts = term_sentence_vector_tf.data
    bm25_sentence_term_idf = TfidfVectorizer()
    term_sentence_vector = bm25_sentence_term_idf.fit_transform([s])
    bm25_idfs = term_sentence_vector.data
    n_features = len(bm25_sentence_term_idf.get_feature_names_out())
    bm25_sentence = [(bm25_idfs[i] * (2.2 * bm25_counts[i])) / (bm25_counts[i] + 1.2 * (0.25 + 0.75 * (n_features) / avg_dl)) for i in range(n_features)]
    term_sentence_dict = dict(zip(bm25_sentence_term_idf.get_feature_names_out(),bm25_sentence))

    cos_similarity_bm25 = cosine_similarity(term_sentence_dict=term_sentence_dict, term_document_dict=term_document_dict,sentence_feature_names=bm25_sentence_term_idf.get_feature_names_out())
    
    return [sentence_count, cos_similarity_tf, cos_similarity_tfidf,cos_similarity_bm25]

def classify(s, d, M, sentence_position,num_sentences):
    prob = M.predict_proba([feature_extraction(s,d,sentence_position,num_sentences)])
    return [prob[:,1],prob[:,1]] 

def evaluation(D,S,M,rf=False,P=[4,6,8,10,12,14,16],L=[500,700,800,1000,1500,2000]):
    map_for_p_docs = [0 for i in range(len(P))]
    std_for_p_docs = []
    map_for_l_docs = [0 for i in range(len(L))]
    count = 1
    if rf:
        count = 1000000
    print("Evaluating Classification Results for Test Set...")
    for doc,summary in tqdm(zip(D,S),total=len(D)):
        if not rf:
            produced_summary_sentences = get_full_summary(doc,M)
        else:
            produced_summary_sentences = get_full_summary_rf(doc,summary,M)
        ref_summary_sentences = sent_tokenize(summary)
        map_for_p, map_for_l = summary_size_evaluation(produced_summary_sentences,ref_summary_sentences,P,L,count)
        for i in range(len(map_for_p)): 
            map_for_p_docs[i] += map_for_p[i]
            if i == 4:
                std_for_p_docs.append(map_for_p[i])
        for i in range(len(map_for_l)): 
            map_for_l_docs[i] += map_for_l[i]
        count+=1
    for i in range(len(map_for_p_docs)): 
        map_for_p_docs[i] = map_for_p_docs[i] / len(D)
    if not rf:
        plot_map_variation(map_for_p_docs, "p_value", P, "general")
    else:
        plot_map_variation(map_for_p_docs, "p_value", P, "general_RF")
    for i in range(len(map_for_l_docs)): 
        map_for_l_docs[i] = map_for_l_docs[i] / len(D)
    if not rf:
        plot_map_variation(map_for_l_docs, "l_value", L, "general")
    else:
        plot_map_variation(map_for_l_docs, "l_value", L, "general_RF")
    average_p_precision = map_for_p_docs[1]
    std_dv_map = statistics.stdev(std_for_p_docs)
    return average_p_precision, std_dv_map