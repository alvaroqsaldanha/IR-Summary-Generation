import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram
import os  
from tqdm import tqdm
import operator
from sklearn.feature_extraction import text
import itertools

from zmq import EVENT_CLOSE_FAILED
from eval import evaluation, get_full_summary, feature_extraction, summary_size_evaluation,plot_map_variation, get_full_summary,get_full_summary_rf
from pagerank import graph, undirected_page_rank

try:
    nltk.data.find('stopwords')
except LookupError:  
    nltk.download('stopwords')
try: 
    nltk.data.find('punkt')
except LookupError: 
    nltk.download('punkt')

news_articles = []
processed_news_articles = []
clean_news_articles = []
clean_news_summaries = []
true_labels = []
reference_summaries = []

### AUXILIARY FUNCTIONS ###

def file_body_processing(filebody): 
    filebody = re.sub(r"[\n]+[.]",'. ',filebody)
    filebody = re.sub(r"[:][-][')']",'',filebody)    
    filebody = re.sub(r"[\'][\']", '\"', filebody)
    filebody = re.sub(r"S.T.A.L.K.E.R.",'',filebody)
    filebody = re.sub(r"[' '][\n]+",'\n',filebody)
    filebody = re.sub(r"[\"][\n]+",'\" ',filebody)
    filebody = re.sub(r"[!][\n]+",'! ',filebody)
    filebody = re.sub(r"[?][\n]+",'? ',filebody)
    filebody = re.sub(r"[:][\n]+",': ',filebody)
    filebody = re.sub(r"[.][\n]+",'. ',filebody)
    filebody = re.sub(r"[\n]+",'. ',filebody)
    return filebody

def read_news_articles(): 
    articles_path = os.walk('BBC News Summary/News Articles')
    print("Reading the news Articles...")
    for article in articles_path: 
        for name in article[2]:
            try: 
                f = open(os.path.join(article[0],name),"r")
                news_article = f.read()
                news_articles.append(str(news_article))
                f.close()
            except: 
                f = open(os.path.join(article[0],name),'rb')
                news_article = f.read()
                news_articles.append(str(news_article))
                f.close()

def process_news_articles(): 
    print("Processing the news articles...")
    for article in tqdm(news_articles, colour='green'): 
        article = file_body_processing(article)
        article = article.lower()
        clean_news_articles.append(article)

def process_news_summaries(): 
    print("Processing news summaries...")
    for summary in tqdm(reference_summaries,colour='cyan'):
        clean_summary = file_body_processing(summary)
        clean_summary = clean_summary.lower()
        summarybody = re.sub(r'([a-z])\.([a-z])', r'\1. \2',clean_summary)
        summarybody = re.sub(r'([0-9])\.([a-z])', r'\1. \2',summarybody)
        summarybody = re.sub(r'([a-z])\.([0-9])', r'\1. \2',summarybody)
        clean_news_summaries.append(summarybody)

def read_reference_summaries(): 
    summaries_path = os.walk('BBC News Summary/Summaries')
    print("Reading the news summaries...")
    for dir_name, subdir_list, files in summaries_path: 
        for name in files: 
            try: 
                f = open(os.path.join(dir_name,name),'r')
                summary = f.read()
                reference_summaries.append(str(summary))
                f.close()
            except: 
                f = open(os.path.join(dir_name,name),'rb')
                summary = f.read()
                reference_summaries.append(str(summary))
                f.close()

def get_tfidf_vectors():
    tfidf = TfidfVectorizer()
    tfidf_vectors = tfidf.fit_transform(clean_news_articles)
    return tfidf_vectors


### CLUSTERING ###

def k_means_study(article_vectors): 
    SSD = []
    print("Studying k-means for varying k...")
    for k in tqdm(range(2,10), colour = 'blue'):
        km = KMeans(n_clusters=k)
        km = km.fit(article_vectors)
        SSD.append(km.inertia_)
    plt.figure(figsize=(10,8))
    plt.title("Plot to select optimal K for clustering")
    plt.xlabel("K number")
    plt.ylabel("SSD")
    plt.plot(range(2,10),SSD,'bx-')
    plt.show()
    plt.clf()

def agglomerative_clustering(article_vectors, num_clusters, linkage, affinity):
    clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage,affinity=affinity,compute_distances=True)
    clustering = clustering.fit(article_vectors.toarray())
    return clustering.labels_

def study_agglomerative_clustering(article_vectors, metric):
    silhouette_scores = []
    print("Studying agglomerative clustering for single linkage and " + metric  + " metric...")
    for k in tqdm(range(2,10), colour = 'green'): 
        labels = agglomerative_clustering(article_vectors, k, 'single', metric)
        silhouette_scores.append(silhouette_score(article_vectors, labels, metric=metric))
    plt.figure(figsize=(10,8))
    plt.title("Plot of single linkage silhouette scores for varying number of clusters")
    plt.xlabel("Num Clusters")
    plt.ylabel("Silhouette Score")
    plt.plot(range(2,10),silhouette_scores)
    plt.savefig("single-linkage-silhouette-"+metric)
    plt.clf()
    max_score = max(silhouette_scores)
    max_score_index = silhouette_scores.index(max_score)
    max_single_linkage = (max_score_index, max_score)
    silhouette_scores = []
    print("Studying agglomerative clustering for complete linkage and " + metric + " metric...")
    for k in tqdm(range(2,10), colour='cyan'): 
        labels = agglomerative_clustering(article_vectors, k, 'complete',metric)
        silhouette_scores.append(silhouette_score(article_vectors, labels, metric=metric))
    plt.figure(figsize=(10,8))
    plt.title("Plot of complete linkage silhouette scores for varying number of clusters")
    plt.xlabel("Num Clusters")
    plt.ylabel("Silhouette Score")
    plt.plot(range(2,10),silhouette_scores)
    plt.savefig("complete-linkage-silhouette-"+metric)
    plt.clf()
    #plt.show()
    max_score = max(silhouette_scores)
    max_score_index = silhouette_scores.index(max_score)
    max_complete_linkage = (max_score_index, max_score)
    silhouette_scores = []
    print("Studying agglomerative clustering for average linkage and " + metric + " metric...")
    for k in tqdm(range(2,10), colour='magenta'): 
        labels = agglomerative_clustering(article_vectors, k, 'average',metric)
        silhouette_scores.append(silhouette_score(article_vectors, labels, metric=metric))
    plt.figure(figsize=(10,8))
    plt.title("Plot of average linkage silhouette scores for varying number of clusters")
    plt.xlabel("Num Clusters")
    plt.ylabel("Silhouette Score")
    plt.plot(range(2,10),silhouette_scores)
    plt.savefig("average-linkage-silhouette-"+metric)
    plt.clf()
    #plt.show()
    max_score = max(silhouette_scores)
    max_score_index = silhouette_scores.index(max_score)
    max_average_linkage = (max_score_index, max_score)
    return max_complete_linkage, max_single_linkage, max_average_linkage

def clustering(D): 
    scores = []
    read_news_articles()
    process_news_articles()
    article_vectors = get_tfidf_vectors()    
    k_means_study(article_vectors)
    best_k = input("Select the best k from the elbow method:")
    km = KMeans(n_clusters=int(best_k))
    km = km.fit(article_vectors)
    score = silhouette_score(article_vectors, km.labels_, metric='euclidean')
    scores.append(score)
    print("The silhouette_score for the best KMeans clustering is: " + str(score))
    max_complete_linkage_cos, max_single_linkage_cos, max_average_linkage_cos = study_agglomerative_clustering(article_vectors, metric = 'cosine')
    scores.append(max_single_linkage_cos[1])
    scores.append(max_complete_linkage_cos[1])
    scores.append(max_average_linkage_cos[1])
    print("The maximum silhouette_score for complete linkage happened with " + str(max_complete_linkage_cos[0] + 3) + " clusters. Silhouette value " + str(max_complete_linkage_cos[1]))
    print("The maximum silhouette_score for single linkage happened with " + str(max_single_linkage_cos[0] + 3) + " clusters. Silhouette value " + str(max_single_linkage_cos[1]))
    print("The maximum silhouette_score for average linkage happened with " + str(max_average_linkage_cos[0] + 3) + " clusters. Silhouette value " + str(max_average_linkage_cos[1]))
    max_complete_linkage, max_single_linkage, max_average_linkage = study_agglomerative_clustering(article_vectors, metric = 'euclidean')
    scores.append(max_single_linkage[1])
    scores.append(max_complete_linkage[1])
    scores.append(max_average_linkage[1])
    print("The maximum silhouette_score for complete linkage happened with " + str(max_complete_linkage[0] + 3) + " clusters. Silhouette value " + str(max_complete_linkage[1]))
    print("The maximum silhouette_score for single linkage happened with " + str(max_single_linkage[0] + 3) + " clusters. Silhouette value " + str(max_single_linkage[1]))
    print("The maximum silhouette_score for average linkage happened with " + str(max_average_linkage[0] + 3) + " clusters. Silhouette value " + str(max_average_linkage[1]))
    print("Studying DBSCAN...")
    dbscan= DBSCAN(eps = 1.25 , min_samples = 25)
    labels = dbscan.fit_predict(article_vectors.toarray())
    score = silhouette_score(article_vectors,labels,metric= 'euclidean')
    print("The silhouette_score for DBSCAN is " + str(score))
    scores.append(score)
    max_score = max(scores)
    max_score_index = scores.index(max_score)
    if max_score_index == 0: 
        print("Chosen clustering solution is KMean for " + str(best_k) + "clusters")
        return  KMeans(n_clusters=int(best_k))
    elif max_score_index == 1: 
        print("Chosen clustering solution is Agglomerative Clustering using single linkage and cosine distance for " + str(max_single_linkage_cos[0] + 3) + " clusters.")
        return  AgglomerativeClustering(n_clusters=max_single_linkage_cos[0] + 3, linkage='single',affinity='cosine',compute_distances=True)
    elif max_score_index == 2:
        print("Chosen clustering solution is Agglomerative Clustering using complete linkage and cosine distance for " + str(max_complete_linkage_cos[0] + 3) + " clusters.")
        return  AgglomerativeClustering(n_clusters=max_complete_linkage_cos[0] + 3, linkage='complete',affinity='cosine',compute_distances=True)
    elif max_score_index == 3: 
        print("Chosen clustering solution is Agglomerative Clustering using average linkage and cosine distance for " + str(max_average_linkage_cos[0] + 3) + " clusters.")
        return  AgglomerativeClustering(n_clusters=max_average_linkage_cos[0] + 3, linkage='complete',affinity='cosine',compute_distances=True)
    elif max_score_index == 4: 
        print("Chosen clustering solution is Agglomerative Clustering using single linkage and euclidean distance for " + str(max_single_linkage[0] + 3) + " clusters.")
        return  AgglomerativeClustering(n_clusters=max_single_linkage[0] + 3, linkage='single',affinity='euclidean',compute_distances=True)
    elif max_score_index == 5:
        print("Chosen clustering solution is Agglomerative Clustering using complete linkage and cosine distance for " + str(max_complete_linkage[0] + 3) + " clusters.")
        return  AgglomerativeClustering(n_clusters=max_average_linkage[0] + 3, linkage='complete',affinity='euclidean',compute_distances=True)
    elif max_score_index == 6: 
        print("Chosen clustering solution is Agglomerative Clustering using average linkage and cosine distance for " + str(max_average_linkage[0] + 3) + " clusters.")
        return  AgglomerativeClustering(n_clusters=max_average_linkage[0] + 3, linkage='complete',affinity='euclidean',compute_distances=True)
    elif max_score_index == 7: 
        print("Chosen clustering solution is DBSCAN")
        return DBSCAN(eps = 1.25, min_samples = 25)

### CLUSTERING INTERPRETING ###

def interpret(cluster,D):
    to_interpret = []
    my_stop_words = text.ENGLISH_STOP_WORDS
    for doc in cluster[0]:
        to_interpret.append(clean_news_articles[doc])
    tfidf = TfidfVectorizer(stop_words=my_stop_words)
    to_interpret_vectors = tfidf.fit_transform(to_interpret)
    sums = to_interpret_vectors.sum(axis=0)
    results_dict = {}
    for col, term in enumerate(tfidf.get_feature_names_out()):
        results_dict[term] = sums[0,col]
    results_dict = dict(sorted(results_dict.items(), key=operator.itemgetter(1),reverse=True))
    return results_dict

def interpret_helper(clustering_solution):
    article_vectors = create_tfidf_vectors() 
    clustering = clustering_solution.fit(article_vectors.toarray())
    SELECTED_CLUSTER = 0
    cluster = [np.where(clustering.labels_ == SELECTED_CLUSTER)[0]]
    scores = interpret(cluster,None)
    scores = dict(itertools.islice(scores.items(), 12))
    plt.bar(scores.keys(), scores.values(), color ='maroon', width = 1)
    plt.xlabel("Term")
    plt.ylabel("Relevance")
    plt.title("Term/Relevance for cluster ." + str(SELECTED_CLUSTER))
    plt.show()

### CLUSTERING EVALUATION ###

def plot_dendrogram(model, **kwargs): 
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_): 
        current_count = 0
        for child_idx in merge: 
            if child_idx < n_samples: current_count += 1
            else: current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_ , counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

def get_true_labels(): 
    articles_path = 'BBC News Summary/News Articles'
    labels = []
    for label in os.listdir(articles_path):
        label_path = os.path.join(articles_path,label)
        for file_path in os.listdir(label_path): 
            labels.append(label_path.split('/')[-1])
    return labels

def evaluate(D, clustering_solution): 
    article_vectors = get_tfidf_vectors() 
    true_labels = get_true_labels()
    clustering = clustering_solution.fit(article_vectors.toarray())
    print("The silhouette score for the chosen clustering solution is: " + str(silhouette_score(article_vectors,clustering.labels_,metric='cosine')))
    print("Plotting the learned dendrogram...")
    plot_dendrogram(clustering, truncate_mode = 'level', p = 5)
    plt.show()
    print("The adjusted rand index for the chosen solution is: " + str(adjusted_rand_score(clustering.labels_,true_labels)))

### CLASSIFICATION ###

def training(Dtrain, Rtrain, model): 
    true_labels = []
    features = []
    print("True Label and Feature Vector Generation...")
    for article, summary in tqdm(zip(Dtrain,Rtrain),total=len(Dtrain)):
        article_sentences = sent_tokenize(article) 
        summary_sentences = sent_tokenize(summary)
        sentence_count = 0
        for article_sentence in article_sentences: 
            sentence_count+=1
            if article_sentence in summary_sentences: 
                true_labels.append(True)
            else: 
                true_labels.append(False)
            features.append(feature_extraction(article_sentence, article, sentence_count,len(article_sentences)))
    if model == 'KNN': 
        neigh = KNeighborsClassifier(n_neighbors=7)
        return neigh.fit(features,true_labels)
    elif model == 'naive_bayes':
        gnb = CategoricalNB()
        return gnb.fit(features,true_labels)

def ranking_extension(d,M):
    article_sentences = sent_tokenize(d)
    sentence_dict = {}
    count = 0
    for sentence in article_sentences:
        count += 1
        sentence_dict[sentence] = classify(sentence,d,M,count,len(article_sentences))[0]
    return dict(sorted(sentence_dict.items(), key=operator.itemgetter(1),reverse=True))

def evaluateClassifier(Dtest,Rtest,M):
    # Without relevance feedback
    print("Starting evaluation of classifier without relevance feedback...")
    #evaluation(Dtest,Rtest,M,rf=False)
    print("Starting evaluation of classifier with relevance feedback...")
    # With relevance feedback
    evaluation(Dtest,Rtest,M,rf=True)

"""
clustering_solution = clustering(None)
evaluate(None, clustering_solution)
"""

def evaluation_page_rank(D,S,P=[4,6,8,10,12,14,16],L=[500,700,800,1000,1500,2000]):
    map_for_p_docs = [0 for i in range(len(P))]
    std_for_p_docs = []
    map_for_l_docs = [0 for i in range(len(L))]
    count = 1
    print("Evaluating Classification Results for Test Set...")
    for doc,summary in tqdm(zip(D,S),total=len(D)):
        produced_summary_sentences = list(undirected_page_rank(doc,16,"cosine_tfidf",0.1).keys())
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
    plot_map_variation(map_for_p_docs, "p_value", P, "general")
    for i in range(len(map_for_l_docs)): 
        map_for_l_docs[i] = map_for_l_docs[i] / len(D)
    plot_map_variation(map_for_l_docs, "l_value", L, "general")
    average_p_precision = map_for_p_docs[1]
    std_dv_map = statistics.stdev(std_for_p_docs)
    return average_p_precision, std_dv_map

def evaluate_page_rank():
    print("Starting evaluation of page rank...")
    evaluation_page_rank(clean_news_articles[2000:2225],clean_news_summaries[2000:2225])   

def build_summary(sentences,l):
  summary = ""
  char_count = 0
  sentence_count = 0
  for entry in sentences:
      char_count += len(entry)
      if char_count < l:
          summary += entry + '\n'
          sentence_count += 1
      else:
          break
  return summary,sentence_count

read_news_articles()
process_news_articles()
read_reference_summaries()
process_news_summaries()

ipt = ''
while ipt != 'q':
    print("\nType and enter:")
    print("1 - to get summaries for a specific file")
    print("2 - to perform clustering study")
    print("3 - to perform full page rank evaluation (best results)")
    print("q - to exit")
    ipt = input('>>> ')
    if ipt == '1':
        m = input("Introduce the model you want to test: (KNN,naive_bayes,PR)")
        file = int(input("Introduce the file you want to summarize (1 to 2225)"))
        if m == "KNN":
            model = training(clean_news_articles[:1000], clean_news_summaries[:1000], model = 'KNN')
            rf = bool(int(input("Use relevance feedback? (1 or 0)")))
            if rf:
                with_rf = get_full_summary_rf(clean_news_articles[file],model)
                print("Summary with relevance feedback:")
                print(build_summary(with_rf,800)[0])
            else:
                without_rf = get_full_summary(clean_news_articles[file],model)
                print("Summary without relevance feedback:")
                print(build_summary(without_rf,800)[0])
        elif m == "naive_bayes":
            rf = bool(int(input("Use relevance feedback? (1 or 0)")))
            model = training(clean_news_articles[:1000], clean_news_summaries[:1000], model = 'naive_bayes')
            if rf:
                with_rf = get_full_summary_rf(clean_news_articles[file],model)
                print("Summary with relevance feedback:")
                print(build_summary(with_rf,800)[0])
            else:
                without_rf = get_full_summary(clean_news_articles[file],model)
                print("Summary without relevance feedback:")
                print(build_summary(without_rf,800)[0])        
        else:
            print("Summary using undirected page rank:")
            print(build_summary(list(undirected_page_rank(clean_news_articles[file],8,"cosine_tfidf",0.3).keys()),800)[0])
    elif ipt == '2':
        clustering_solution = clustering(None)
        interpret_helper(clustering_solution)
        evaluate(None, clustering_solution)
    elif ipt == '3': 
        evaluate_page_rank()
    elif ipt == 'q':
        exit()