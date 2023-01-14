# Summary Generation - Information Retrieval

This project is a summary generation tool, using the [BBC News Summary dataset](https://www.kaggle.com/c/learn-ai-bbc). Given a document, it aims to choose its most relevant senteces using both unsupervised and supervised methods, returning a summary.

For unsupervised summary generation, an inverted index architecture is used (indexing also biwords, noun phrases, ...) with multiple potential relevance measures like typical Term Frequency, TFIDF (Term Frequency with Inverse Document Frequency), and BM-25.
Other performance improving mechanisms are implemented such as reciprocal rank fusion and maximal marginal relevance.
The results are then evaluated using the given reference summaries.

A demo is provided [here](https://github.com/alvaroqsaldanha/Information-Retrieval-Summary-Generation/blob/main/Unsupervised%20Summary%20Generation/demo_notebook.ipynb).

For supervised summary generation, machine learning models (more specifically K-Nearest Neighbours, Naive Bayes, and Neural Networks) are used to classify a specific sentence as relevant or not, depending on its presence in the reference summary. Also for relevance classification, the page rank algorithm is implemented and tested. In-depth feature engineering (sentence input), relevance feedback techniques, and clustering analysis of document similarity and category through unsupervised learning algorithms are also explored.

A demo is provided [here](https://github.com/alvaroqsaldanha/Information-Retrieval-Summary-Generation/blob/main/Supervised%20Summary%20Generation/demo_notebook.ipynb).
