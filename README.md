# Summary Generation - Information Retrieval

This project is a summary generation tool, using the [BBC News Summary dataset](https://www.kaggle.com/c/learn-ai-bbc). Given a document, it aims to choose its most relevant senteces using both unsupervised and supervised methods, returning a summary.

For unsupervised summary generation, an inverted index architecture is used (indexing also biwords, noun phrases, ...) with multiple potential relevance measures like typical Term Frequency, TFIDF (Term Frequency with Inverse Document Frequency), and BM-25.
Other performance improving mechanisms are implemented such as reciprocal rank fusion and maximal marginal relevance.
The results are then evaluated using the given reference summaries.

A demo is provided [here](https://github.com/alvaroqsaldanha/Information-Retrieval-Summary-Generation/blob/main/Unsupervised%20Summary%20Generation/demo_notebook.ipynb).

For supervised summary generation, machine learning models (more specifically K-Nearest Neighbours, Naive Bayes, and Neural Networks) are used to classify a specific sentence as relevant or not, depending on its presence in the reference summary. Also for relevance classification, the page rank algorithm is implemented and tested. In-depth feature engineering (sentence input), relevance feedback techniques, and clustering analysis of document similarity and category through unsupervised learning algorithms are also explored.

A demo is provided [here](https://github.com/alvaroqsaldanha/Information-Retrieval-Summary-Generation/blob/main/Supervised%20Summary%20Generation/demo_notebook.ipynb).

## Unsupervised Summary Generation Example

### Original Text:

```
Ink helps drive democracy in Asia
The Kyrgyz Republic, a small, mountainous state of the former Soviet republic, is using invisible ink and ultraviolet readers in the country's elections as part of a drive to prevent multiple voting.
This new technology is causing both worries and guarded optimism among different sectors of the population. In an effort to live up to its reputation in the 1990s as "an island of democracy", the Kyrgyz President, Askar Akaev, pushed through the law requiring the use of ink during the upcoming Parliamentary and Presidential elections. The US government agreed to fund all expenses associated with this decision.
The Kyrgyz Republic is seen by many experts as backsliding from the high point it reached in the mid-1990s with a hastily pushed through referendum in 2003, reducing the legislative branch to one chamber with 75 deputies. The use of ink is only one part of a general effort to show commitment towards more open elections - the German Embassy, the Soros Foundation and the Kyrgyz government have all contributed to purchase transparent ballot boxes.
The actual technology behind the ink is not that complicated. The ink is sprayed on a person's left thumb. It dries and is not visible under normal light.
However, the presence of ultraviolet light (of the kind used to verify money) causes the ink to glow with a neon yellow light. At the entrance to each polling station, one election official will scan voter's fingers with UV lamp before allowing them to enter, and every voter will have his/her left thumb sprayed with ink before receiving the ballot. If the ink shows under the UV light the voter will not be allowed to enter the polling station. Likewise, any voter who refuses to be inked will not receive the ballot. These elections are assuming even greater significance because of two large factors - the upcoming parliamentary elections are a prelude to a potentially regime changing presidential election in the Autumn as well as the echo of recent elections in other former Soviet Republics, notably Ukraine and Georgia. The use of ink has been controversial - especially among groups perceived to be pro-government.
Widely circulated articles compared the use of ink to the rural practice of marking sheep - a still common metaphor in this primarily agricultural society.
The author of one such article began a petition drive against the use of the ink. The greatest part of the opposition to ink has often been sheer ignorance. Local newspapers have carried stories that the ink is harmful, radioactive or even that the ultraviolet readers may cause health problems. Others, such as the aggressively middle of the road, Coalition of Non-governmental Organizations, have lauded the move as an important step forward. This type of ink has been used in many elections in the world, in countries as varied as Serbia, South Africa, Indonesia and Turkey. The other common type of ink in elections is indelible visible ink - but as the elections in Afghanistan showed, improper use of this type of ink can cause additional problems. The use of "invisible" ink is not without its own problems. In most elections, numerous rumors have spread about it.
In Serbia, for example, both Christian and Islamic leaders assured their populations that its use was not contrary to religion. Other rumours are associated with how to remove the ink - various soft drinks, solvents and cleaning products are put forward. However, in reality, the ink is very effective at getting under the cuticle of the thumb and difficult to wash off. The ink stays on the finger for at least 72 hours and for up to a week. The use of ink and readers by itself is not a panacea for election ills. The passage of the inking law is, nevertheless, a clear step forward towards free and fair elections." The country's widely watched parliamentary elections are scheduled for 27 February.
David Mikosz works for the IFES, an international, non-profit organisation that supports the building of democratic societies.
```

### Summary

Using TF-IDF:

```
Ink helps drive democracy in Asia. The ink is sprayed on a person's left thumb. 
Widely circulated articles compared the use of ink to the rural practice of marking sheep - a still common metaphor in this primarily agricultural society. 
The Kyrgyz Republic, a small, mountainous state of the former Soviet republic, is using invisible ink and ultraviolet readers in the country's elections as part of a drive to prevent multiple voting.
```

Using Reciprocal Rank Fusion:

```
Ink helps drive democracy in Asia.The ink is sprayed on a person's left thumb. 
The use of ink and readers by itself is not a panacea for election ills.The Kyrgyz Republic, a small, mountainous state of the former Soviet republic, is using invisible ink and ultraviolet readers in the country's elections as part of a drive to prevent multiple voting. 
The use of "invisible" ink is not without its own problems. Widely circulated articles compared the use of ink to the rural practice of marking sheep - a still common metaphor in this primarily agricultural society. Local newspapers have carried stories that the ink is harmful, radioactive or even that the ultraviolet readers may cause health problems. 
If the ink shows under the UV light the voter will not be allowed to enter the polling station. Likewise, any voter who refuses to be inked will not receive the ballot. It dries and is not visible under normal light.The country\'s widely watched parliamentary elections are scheduled for 27 February.
```

Using Maximal Marginal Relevance:

```
'Ink helps drive democracy in Asia.'
"The Kyrgyz Republic, a small, mountainous state of the former Soviet republic, is using invisible ink and ultraviolet readers in the country's elections as part of a drive to prevent multiple voting."
'This new technology is causing both worries and guarded optimism among different sectors of the population.'
````

## Supervised Summary Generation Example

### Original Text:

```
Ad sales boost Time Warner profit
Quarterly profits at US media giant TimeWarner jumped 76% to $1.13bn (£600m) for the three months to December, from $639m year-earlier.
The firm, which is now one of the biggest investors in Google, benefited from sales of high-speed internet connections and higher advert sales. TimeWarner said fourth quarter sales rose 2% to $11.1bn from $10.9bn. Its profits were buoyed by one-off gains which offset a profit dip at Warner Bros, and less users for AOL.
Time Warner said on Friday that it now owns 8% of search-engine Google. But its own internet business, AOL, had has mixed fortunes. It lost 464,000 subscribers in the fourth quarter profits were lower than in the preceding three quarters. However, the company said AOL's underlying profit before exceptional items rose 8% on the back of stronger internet advertising revenues. It hopes to increase subscribers by offering the online service free to TimeWarner internet customers and will try to sign up AOL's existing customers for high-speed broadband. TimeWarner also has to restate 2000 and 2003 results following a probe by the US Securities Exchange Commission (SEC), which is close to concluding.
Time Warner's fourth quarter profits were slightly better than analysts' expectations. But its film division saw profits slump 27% to $284m, helped by box-office flops Alexander and Catwoman, a sharp contrast to year-earlier, when the third and final film in the Lord of the Rings trilogy boosted results. For the full-year, TimeWarner posted a profit of $3.36bn, up 27% from its 2003 performance, while revenues grew 6.4% to $42.09bn. "Our financial performance was strong, meeting or exceeding all of our full-year objectives and greatly enhancing our flexibility," chairman and chief executive Richard Parsons said. For 2005, TimeWarner is projecting operating earnings growth of around 5%, and also expects higher revenue and wider profit margins.
TimeWarner is to restate its accounts as part of efforts to resolve an inquiry into AOL by US market regulators. It has already offered to pay $300m to settle charges, in a deal that is under review by the SEC. The company said it was unable to estimate the amount it needed to set aside for legal reserves, which it previously set at $500m. It intends to adjust the way it accounts for a deal with German music publisher Bertelsmann's purchase of a stake in AOL Europe, which it had reported as advertising revenue. It will now book the sale of its stake in AOL Europe as a loss on the value of that stake.
```

### Summary

Using KNN:

```
michael howard has finally revealed the full scale of his planned tory tax cuts.
at its simplest, it is saying: "vote tory and you can have it both ways".
should he win the next general election, he has earmarked £4 billion that will be used to reduce taxes - although he still will not say which or how.
and even after that was done, it would still have enough left over for a tax cut equivalent to about a penny off the basic rate of income tax.
it is a move back towards an almost traditional tory message which previously suggested labour was the party of tax rises and the conservatives the party of tax cuts.
the extension of that, however, was that labour was also seen as the party of big spending on the public services while the tories were the cutters.
```

Using KNN with Ranking Extension:

```
michael howard has finally revealed the full scale of his planned tory tax cuts.
at its simplest, it is saying: "vote tory and you can have it both ways".
should he win the next general election, he has earmarked £4 billion that will be used to reduce taxes - although he still will not say which or how.
and even after that was done, it would still have enough left over for a tax cut equivalent to about a penny off the basic rate of income tax.
it is a move back towards an almost traditional tory message which previously suggested labour was the party of tax rises and the conservatives the party of tax cuts.
the extension of that, however, was that labour was also seen as the party of big spending on the public services while the tories were the cutters.
```

Using Page Rank:

```
not only would his government stick to labour spending plans on core public services, including health and education, it would increase spending on defence, police and pensions.
at its simplest, it is saying: "vote tory and you can have it both ways".
this was the pre-election message many in his party have been pressing for and voters, he believes, will warm to.
should he win the next general election, he has earmarked £4 billion that will be used to reduce taxes - although he still will not say which or how.
he insists he will not promise anything before the election that he cannot deliver if put into downing street.
````

