Hive: News Aggregation Model

Hive is a tool developed to identify, cluster, and summarize trending socio-political discourse from Reddit 
over the past week. It uses natural language processing (NLP), unsupervised clustering, and OpenAI generation 
news headlines and summaries.


Goals

1. Collect Reddit posts from a curated list of socio-political subreddits over the last 7 days.
2. Vectorize post text to make the data interpretable to the classifier
3. Filter for relevance using a binary classifier trained to detect socio-political content.
4. Embed, cluster, and summarize key themes using sentence embeddings and HDBSCAN.
5. Present results as professional, unbiased news briefings via a Gradio interface


Requirements

The Hive program uses:
  * numpy 
  * spacy 
  * hdbscan
  * praw 
  * openai 
  * sentence-transformers 
  * pandas 
  * joblib
  * scipy 
  * gradio


Model Components

AutoVectorizer: A trained vectorizer model that transforms input strings into vectors, capturing the strings’ 
                textual features. The vector form can be interpreted by the AutoClassifier.

AutoClassifier: A binary classification model that labels vectorized Reddit posts as either sociopolitical (1) or
                not (0). It is used to filter out any irrelevant posts from the data set.


Main Script (app.py)

1. Data Collection
  * Pulls post titles (text) from specified subreddits using praw
  * Filters by timestamp to ensure posts are within the last 7 days

2. Preprocessing and Filtering
  * Converts post titles to feature vectors using AutoVectorizer
  * Pads feature matrices if dimensions are <5000 (model requirement)
  * Classifies posts using AutoClassifier and filters for socio-political posts
  * Removes special characters, digits, and extra whitespace

3. Embedding & Clustering
  * Uses SentenceTransformers (all-MiniLM-L6-v2) to embed post titles into 384-D semantic vectors.
  * Clusters embeddings using HDBSCAN (min_cluster_size=20) to identify themes
  * Labels with under 20 instances are assigned as noise (label = -1)

4. Post-Clustering Analysis
  * Groups cluster texts into a dictionary: { cluster_id: [text1, text2, ...] }
  * Counts number of proper nouns in each cluster as an approximation of newsworthiness
  * Ranks clusters by proper noun count

5. GenAI-based Summarization
  * Generates formatted headings and summaries for each cluster’s posts using gpt-4o

6. Interfacing
  * Wraps all major script processes into the summarize_clusters_wrapper function
  * Gradio UI calls the wrapper function and integrates the back-end and front-end


Helper Functions

* fetch_posts - Pulls Reddit posts by subreddit and timestamp
* simple_preprocess - Cleans raw text using regex
* generate_header - Generates an AP-style headline for a given cluster
* generate_summary - Generates summaries for a cluster using GPT
* naive_count_proper_nouns - Counts proper nouns to approximate newsworthiness
* get_word_frequencies - Analyzes term frequency excluding stopwords
* summarize_clusters - Facilitates the heading and summary generation
* summarize_clusters_wrapper - Wrapper function for Gradio UI integration


End Result

A web app that allows you to generate professional-quality news stories, after summarizing the socio-political 
discourse on Reddit over the previous week.