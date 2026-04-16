import os, re, datetime
import numpy as np, spacy, hdbscan, praw, openai
from collections import Counter
from sentence_transformers import SentenceTransformer
import datetime
import pandas as pd
import joblib
import scipy.sparse as sp
import gradio as gr
from sklearn.cluster import KMeans
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import traceback
import asyncio
import aiohttp
from typing import List, Tuple


"""**Collect all Reddit data from past week**"""
max_vids = 5
cmv = 5
openai_api_key = os.getenv('open_api')
yt_api=os.getenv('yt_api')
nlp = spacy.load("en_core_web_sm")
ENTITY_LABELS_TO_KEEP = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT", "WORK_OF_ART"}

ENTITY_STOPLIST = {
    "Reddit", "YouTube", "Instagram", "Twitter",
    "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday",
    "Today","Yesterday","Tomorrow"
}

def top_spacy_entities(texts: list[str], top_k=None) -> list[str]:
    """
    MODIFIED: Extracts all individual Nouns and Proper Nouns from the input texts,
    ignoring common stoplist terms. The top_k parameter is now ignored.
    This function will be used to extract terms from the Context input.
    """
    counts = Counter()

    for txt in texts:
        if not isinstance(txt, str) or not txt.strip():
            continue
        doc = nlp(txt)
        
        for token in doc:
            if token.pos_ not in {"NOUN", "PROPN"}:
                continue

            phrase = token.text.strip()
            
            if phrase in ENTITY_STOPLIST:
                continue
            if len(phrase) < 3:
                continue
            
            if token.pos_ == "NOUN":
                 phrase = phrase.lower()

            counts[phrase] += 1

    return [phrase for phrase, _ in counts.most_common()]

def context_to_hashtags(noun_list: list[str], query: str) -> list[str]:
    """
    MODIFIED: Takes the pre-extracted list of nouns and ensures the original 
    query is included, creating the final set of unique search terms.
    """
    terms = set(noun_list)

    if query.strip():
        terms.add(query.strip()) 

    return list(terms)

# =========================
# Classifier / Vectorizer
# =========================
classifier = joblib.load("AutoClassifier.pkl")
vectorizer = joblib.load("AutoVectorizer.pkl")

NO_POP_FILTER = False

def is_pop_culture_choice(choice: str) -> bool:
    """
    Returns True if the dropdown choice indicates 'Pop culture',
    False if 'Not pop culture' (or anything else).
    """
    if not isinstance(choice, str):
        return True
    return choice.strip().lower().startswith("pop")

# =========================
# Embeddings / OpenAI / Reddit
# =========================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = openai.OpenAI(api_key=openai_api_key)

reddit = praw.Reddit(
    client_id=os.getenv('client_id'),
    client_secret=os.getenv('client_secret'),
    user_agent='HFHIve',
    check_for_async=False
)

end_date = datetime.datetime.utcnow().date()
start_date = end_date - datetime.timedelta(days=14)

start_ts = int(datetime.datetime.combine(start_date, datetime.time.min).timestamp())


def fetch_posts(query, subreddit_name="all", limit=None):
    try:
        all_rows = []
        subreddit = reddit.subreddit(subreddit_name)

        results = subreddit.search(query, limit=limit, sort="hot")

        for post in results:
            all_rows.append({
                "post_id": post.id,
                "parent_post_id": post.id,
                "title": post.title,
                "content": post.selftext,
                "created_utc": post.created_utc,
                "score": post.score,
                "num_comments": post.num_comments,
                "upvote_ratio": post.upvote_ratio,
                "url": post.url,
                "subreddit": post.subreddit.display_name,
                "is_comment": False
            })

        return all_rows

    except Exception as e:
        print(f"Reddit fetch error: {e}")
        return []



lock = threading.Lock()

def fetch_comments_for_video(video_id, published, api_key, comments_per_video):
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": comments_per_video,
        "textFormat": "plainText",
        "key": api_key
    }

    try:
        items = requests.get(url, params=params).json().get("items", [])
    except requests.exceptions.RequestException:
        return []

    rows = []
    for c in items:
        text = c["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        rows.append({"text": text, "date": published})

    return rows


def process_hashtag(tag, api_key, max_videos, comments_per_video, seen):
    local_rows = []

    query = tag.lstrip("#")
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": f"#{query}",
        "type": "video",
        "maxResults": max_videos,
        "key": api_key
    }

    try:
        results = requests.get(url, params=params).json().get("items", [])
    except requests.exceptions.RequestException:
        return []

    futures = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        for item in results:
            vid = item.get("id", {})
            if vid.get("kind") != "youtube#video":
                continue

            video_id = vid.get("videoId")
            if not video_id:
                continue

            with lock:
                if video_id in seen:
                    continue
                seen.add(video_id)

            published = item["snippet"]["publishedAt"]

            futures.append(
                executor.submit(
                    fetch_comments_for_video,
                    video_id,
                    published,
                    api_key,
                    comments_per_video
                )
            )

        for f in as_completed(futures):
            local_rows.extend(f.result())

    return local_rows

def get_youtube_comments_for_hashtags(
    hashtags, max_videos=max_vids, comments_per_video=cmv
):
    api_key = yt_api
    seen = set()
    rows = []

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(
                process_hashtag,
                tag,
                api_key,
                max_videos,
                comments_per_video,
                seen
            )
            for tag in hashtags
        ]

        for f in as_completed(futures):
            rows.extend(f.result())

    if not rows:
        return pd.DataFrame(columns=["text", "date"])

    print(f"{len(rows)} youtube posts collected")
    return pd.DataFrame(rows)


def simple_preprocess(texts):
    """OPTIMIZED: Vectorized preprocessing"""
    if isinstance(texts, list):
        texts = pd.Series(texts)
        texts = texts.str.lower().str.strip()
        texts = texts.str.replace(r'[^a-z0-9\s]+', '', regex=True)
        texts = texts.str.replace(r'\s+', ' ', regex=True)
        return texts.tolist()
    else:
        text = texts.lower().strip()
        text = re.sub(r'[^a-z0-9\s]+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

# =========================
# Filtering helper
# =========================
def filter_texts_for_trend(texts, pop_choice):
    """
    Use the classifier/vectorizer to keep either pop-culture or not-pop-culture
    posts depending on pop_choice. If NO_POP_FILTER is True, returns texts unchanged.
    """
    global NO_POP_FILTER

    if NO_POP_FILTER:
        return texts

    if classifier is None or vectorizer is None or not texts:
        return texts

    pop_culture = is_pop_culture_choice(pop_choice)

    X = vectorizer.transform(texts)
    predictions = classifier.predict(X)
    keep_label = 0 if pop_culture else 1

    filtered = [t for t, pred in zip(texts, predictions) if pred == keep_label]
    return filtered

# =========================
# ASYNC OpenAI API Functions
# =========================

async def generate_summary_async(query, cluster_texts, is_question):
    """ASYNC version of generate_summary - runs concurrently"""
    prompt = ""
    if is_question: 
        prompt = f"""You analyze online chatter and accurately answer questions about themes within social conversations.
        
        Your original query was: "{query}".
            
        You are given a collection of social media excerpts. First, determine whether
        THE MAJORITY of these excerpts meaningfully provide insight on the query "{query}".
        
        If there are very few or NO excerpts that relate to the topics in "{query}", then:
        - Produce a short off-topic report reflecting what the online conversation is actually about.
        - Use EXACTLY this format:
          [Off-topic Summary]
          [Bullet 1: State what the majority of online chatter is focusing on instead, naming the most common off-topic themes, people, or entities.]
          [Bullet 2: 1-3 sentences explaining what, if anything, is said about "{query}".]
        - Keep everything under 80 words total.
        
        If there ARE excerpts relating to themes within "{query}", generate a concise ON-TOPIC trend report using this format:
        
        [Number]. [Concise Title in Title Case]
        [Bullet 1: Provide a concise and direct answer to the query in (1-2 sentences)
        [Bullet 2: Elaborate on the answer provided, mentioning WHY people feel this way, with AT LEAST 1 direct quote from online chatter (2-3 sentences)]
        [Bullet 3: Note on broader impact, different opinions, or disagreements in the online chatter and WHY they are present (3-4 sentences)]
        [Bullet 4: One-sentence outlook describing potential changes in the MAJORITY of perspectives or shifts in the discussion]
    
        ON-TOPIC Requirements:
        1. Keep a neutral, analytical tone (not journalistic).
        2. Each bullet should be one concise sentence (≤ 25 words).
        3. Include at least one relevant, direct quote from the conversation.
        4. Include concrete names, places, or events when available.
        5. Never mention the word "cluster" or refer to data collection methods.
        
        Input excerpts:
        {" ".join(cluster_texts)}
        
        Generate ONLY the final report using the correct format:
        """
    else:
        prompt = f"""You analyze online chatter and identify dominant themes within social conversations.
    
        Your original query/topic was: "{query}".
            
        You are given a collection of social media excerpts. First, determine whether
        THE MAJORITY of these excerpts meaningfully discuss the query/topic "{query}".
        
        If the majority of excerpts are NOT about "{query}", then:
        - Produce a short off-topic report reflecting what the online conversation is actually about.
        - Use EXACTLY this format:
          [Off-topic Summary]
          [Bullet 1: State what the majority of online chatter is focusing on instead, naming the most common off-topic themes, people, or entities.]
          [Bullet 2: One sentence explaining why this group of posts provides little insight into "{query}".]
        - Keep everything under 80 words total.
        
        If the majority of excerpts ARE about "{query}", generate a concise ON-TOPIC trend report using this format:
        
        [Number]. [Concise Title in Title Case]
        [Bullet 1: Key development or repeated pattern in the online conversation with 1–2 specific details or named entities (1-2 sentences)]
        [Bullet 2: Brief context about who is driving the conversation or what groups are involved (1 sentence)]
        [Bullet 3: Note on broader impact, reactions, or disagreements in the online chatter (3-4 sentences)]
        [Bullet 4: One-sentence outlook describing potential follow-up topics or shifts in the discussion]
        
        ON-TOPIC Requirements:
        1. Keep a neutral, analytical tone (not journalistic).
        2. Each bullet should be one concise sentence (≤ 25 words).
        3. Include at least one relevant, direct quote from the conversation.
        4. Include concrete names, places, or events when available.
        5. Never mention the word "cluster" or refer to data collection methods.
        
        Input excerpts:
        {" ".join(cluster_texts)}
        
        Generate ONLY the final report using the correct format:
        """
        
    try:
        # Run the blocking OpenAI call in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.responses.create(
                model="gpt-5.1",
                input=prompt
            )
        )
        return response.output_text
    except Exception as e:
        return f"[OpenAI error during summary generation]: {e}"


async def generate_header_async(query, cluster_texts):
    """ASYNC version of generate_header - runs concurrently"""
    prompt = f"""You analyze online chatter and summarize the dominant theme in a short headline.
    Original query/topic: "{query}".
    
    Step 1: Determine whether the majority of excerpts meaningfully relate to "{query}".
    
    If the majority are NOT about "{query}":
    - Output a brief headline clearly indicating the conversation is off-topic.
    - Example style: "Online Chatter Focuses On X, Not {query}"
    - Keep it under 12–15 words.
    If the majority ARE about "{query}":
    - Create a concise headline that:
      1. Is written in Title Case.
      2. Contains a verb.
      3. Mentions major entities or locations if present.
      4. Reflects the main theme of the online conversation.
      5. Does NOT mention social media platforms, Reddit, or clustering.
      6. Stays under 15 words.
    
    Excerpts:
    {" ".join(cluster_texts)}
    Output ONLY the final headline text:
    """
        
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.responses.create(
                model="gpt-5.1",
                input=prompt
            )
        )
        return response.output_text
    except Exception as e:
        return f"[OpenAI error during headline generation]: {e}"



async def generate_final_report_async(query: str, full_summary_report: str, top_cluster_raw_texts: list[str]) -> str:
    """ASYNC version of generate_final_report"""
    
    deep_dive_context = "\n".join(top_cluster_raw_texts)
    
    prompt = f"""You are an executive trend analyst. Your task is to synthesize 
    a final, comprehensive report from structured trend briefings and a selection of raw social data.
    Original Query/Topic: "{query}"
    ---
    
    PART 1: STRUCTURED TREND BRIEFINGS (Analyze this entire section for all key developments and context)
    {full_summary_report}
    ---
    
    PART 2: RAW DATA CONTEXT (Cross-reference these raw posts for nuanced, controversial, or specific details in the dominant trends)
    {deep_dive_context}
    
    ---
    Generate the FINAL Executive Trend Report, addressing these points:
    
    1.  **Overall Theme and Dominant Narrative (1-2 sentences):** State the main consensus or focus of the entire conversation.
    2.  **Key Controversies/Disagreements (1-2 short paragraphs):** Identify and elaborate on 2-3 significant, controversial, or polarizing takes/arguments that appeared across the top discussions, citing specific entities or terminology from the raw data.
    3.  **Future Outlook (1 sentence):** Conclude with the likely next stage or point of discussion for this topic.
    
    Keep the tone neutral, analytical, and professional. The entire report should be concise, ideally under 250 words.
    """

    try:
        start = time.time()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.responses.create(
                model="gpt-5.1",
                input=prompt
            )
        )
        end = time.time()
        print(f"final report openapi call runtime: {end-start:.2f}s")
        return response.output_text
    except Exception as e:
        return f"[OpenAI error during final report generation]: {e}"


# =========================
# Async cluster processing
# =========================

async def process_cluster_async(idx: int, cid: int, query: str, context_question: str, clusters: dict) -> Tuple[int, str, str]:
    """ASYNC version - runs header and summary generation in parallel"""
    cluster_texts = clusters[cid]
    prompt = context_question or query
    use_context = bool(context_question)

    # Run header and summary generation CONCURRENTLY
    header_task = generate_header_async(prompt, cluster_texts)
    summary_task = generate_summary_async(prompt, cluster_texts, use_context)
    
    header, summary = await asyncio.gather(header_task, summary_task)
    
    return idx, header, summary


async def summarize_clusters_async(query: str, proper_counts: dict, clusters: dict, context_question: str) -> str:
    """ASYNC version - generates all cluster summaries concurrently"""
    if not proper_counts:
        return "No valid clusters found."

    top_clusters = sorted(proper_counts, key=proper_counts.get, reverse=True)
    output = "### Top News Briefings\n\n"
    
    top_k = min(5, len(top_clusters))
    
    # Create tasks for all clusters - they run concurrently
    start = time.time()
    tasks = [
        process_cluster_async(idx, cid, query, context_question, clusters)
        for idx, cid in enumerate(top_clusters[:top_k], 1)
    ]
    
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])
    
    end = time.time()
    print(f"summarized {top_k} clusters concurrently in {end-start:.2f}s")
    
    for idx, header, summary in results:
        output += f"#### {idx}. {header}\n\n"
        output += f"{summary}\n\n---\n\n"
    return output


def naive_count_proper_nouns(texts_list):
    """
    OPTIMIZED: Vectorized proper noun counting
    """
    pattern = re.compile(r'\b[A-Z][a-z]+\b')
    return sum(len(pattern.findall(text)) for text in texts_list)


def get_word_frequencies(texts_list, stopwords=None):
    if stopwords is None:
        stopwords = {"the", "and", "this", "that", "with", "from", "for", "was", "were", "are"}
    all_text = " ".join(texts_list).lower()
    words = re.findall(r'\w+', all_text)
    words = [w for w in words if w not in stopwords]
    return Counter(words)


def summarize_clusters_wrapper(query, context, pop_choice, no_pop_filter, context_question):
    """Wrapper that runs async operations"""
    try:
        global NO_POP_FILTER
        NO_POP_FILTER = bool(no_pop_filter)

        query = (query or "").strip()
        context = (context or "").strip()
        context_question = (context_question or "").strip()

        if not query:
            return "Please enter a word or phrase to search."

        # --- 1) DETERMINE SEARCH TERMS ---
        context_as_list = [context]
        extracted_nouns = top_spacy_entities(context_as_list)

        youtube_search_terms = set(extracted_nouns)
        youtube_search_terms.add(query)
        youtube_search_terms = list(youtube_search_terms)

        # --- 2) REDDIT DATA FETCHING ---
        all_posts = []

        try:
            base_posts = fetch_posts(f"{query} {context}", subreddit_name="all", limit=150)  
            all_posts.extend(base_posts)
        except Exception as e:
            print(f"[WARN] Reddit fetch failed: {e}")

        if not all_posts:
            return f"No Reddit posts found for query: '{query}'."

        posts_df = pd.DataFrame(all_posts)
        posts_df = posts_df.drop_duplicates(subset="post_id")

        posts_df["time"] = pd.to_datetime(posts_df["created_utc"], unit="s")
        seven_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=7)
        posts_df = posts_df[posts_df["time"] >= seven_days_ago]

        reddit_texts = (
            posts_df["title"].fillna("") + " " + posts_df["content"].fillna("")
        ).tolist()

        # --- 3) YOUTUBE DATA FETCHING ---
        youtube_df = get_youtube_comments_for_hashtags(youtube_search_terms)
        youtube_texts = youtube_df["text"].tolist()

        # --- 4) COMBINE ---
        texts = reddit_texts + youtube_texts
        if not texts:
            return f"No social media texts found for query: '{query}'."

        # --- FILTER ---
        filtered_texts = filter_texts_for_trend(texts, pop_choice)

        if not filtered_texts:
            return "No texts matched the selected content filter."
        
        # OPTIMIZED: Deduplicate exact matches before embedding
        filtered_texts = list(dict.fromkeys(filtered_texts))
        
        processed_texts = [simple_preprocess(t) for t in filtered_texts]

        # --- EMBEDDING ---
        embeddings = embed_model.encode(processed_texts, batch_size=64)

        # --- CLUSTERING ---
        n = len(filtered_texts)

        if n < 3:
            clusters = {0: filtered_texts}
        else:
            min_cluster_size = max(3, min(15, n // 10))
            min_samples = max(1, min_cluster_size // 2)

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples
            )

            labels = clusterer.fit_predict(embeddings)

            if (labels == -1).all():
                print("[INFO] HDBSCAN failed → fallback to KMeans")
                k = min(3, n)
                km = KMeans(n_clusters=k, n_init=10, random_state=0)
                labels = km.fit_predict(embeddings)

            clusters = {}
            for text, lbl in zip(filtered_texts, labels):
                clusters.setdefault(lbl, []).append(text)

        # --- SUMMARIZATION ---
        proper_counts = {
            cid: naive_count_proper_nouns(txts)
            for cid, txts in clusters.items()
            if cid != -1
        }

        if not proper_counts:
            clusters = {0: filtered_texts}
            proper_counts = {0: naive_count_proper_nouns(filtered_texts)}

        # --- RUN ASYNC OPERATIONS ---
        # This is where we get the big speedup from concurrent API calls
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            full_summary_report = loop.run_until_complete(
                summarize_clusters_async(query, proper_counts, clusters, context_question)
            )
            
            # --- TOP CLUSTERS ---
            top_cluster_ids = sorted(proper_counts, key=proper_counts.get, reverse=True)
            
            top_k_raw_texts = []
            for cid in top_cluster_ids[:2]:
                header_input = context_question if context_question else query
                header = loop.run_until_complete(
                    generate_header_async(header_input, clusters[cid])
                )
                top_k_raw_texts.append(
                    f"== Raw Context for Theme: {header} ==\n" +
                    "\n".join(clusters[cid][:20])  
                )
            
            # --- FINAL REPORT ---
            report_query = context_question if context_question else query
            final_executive_report = loop.run_until_complete(
                generate_final_report_async(report_query, full_summary_report, top_k_raw_texts)
            )
        finally:
            loop.close()
        
        final_output = f"""
## 🎯 Executive Report: {query.title()}
{final_executive_report}
---
## Detailed Trend Briefings
{full_summary_report}
"""

        return final_output

    except Exception as e:
        error_msg = f"""
### Error occurred
{str(e)}
Traceback:
{traceback.format_exc()}
"""
        print(error_msg)
        return error_msg


# =========================
# Gradio UI
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📰 Hive News Trend Report Generator")

    with gr.Row():
        query_input = gr.Textbox(
            label="1. Primary Topic/Search Phrase",
            placeholder="e.g. Tesla stock, Taylor Swift tour, RTX 5090",
            scale=1
        )
        context_input = gr.Textbox(
            label="2. Context (Optional Search Expansion)",
            placeholder="e.g. Elon Musk, Gigafactory, Cybertruck (for Tesla) — Used for broader search coverage.",
            scale=1
        )

    gr.Markdown("""
    ### 3. Content Filtering
    **Default Recommended:** To capture the broadest view, the **'No pop-culture filter'** box below should remain checked. Only use the content filter for highly focused or more thorough analysis.
    """)

    with gr.Row():
        pop_toggle = gr.Dropdown(
            label="Filter Type (Only used if 'No pop-culture filter' is unchecked)",
            choices=["Pop culture", "Not pop culture"],
            value="Pop culture",
            scale=1
        )
        no_pop_filter_cb = gr.Checkbox(
            label="No pop-culture filter (Recommended for broadest data)",
            value=True,
            scale=1
        )

    with gr.Row():
        context_query = gr.Textbox(
            label = "Query (Optional Search)",
            placeholder = "Do people think Teslas are too expensive?",
            scale=1
        )

    analyze_btn = gr.Button("Cluster + Generate Report", variant="primary")
    output_md = gr.Markdown()

    analyze_btn.click(
        fn=summarize_clusters_wrapper,
        inputs=[query_input, context_input, pop_toggle, no_pop_filter_cb, context_query],
        outputs=output_md
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False
)