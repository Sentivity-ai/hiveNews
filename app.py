import os, re, datetime
import numpy as np, spacy, hdbscan, praw, openai
from collections import Counter
from sentence_transformers import SentenceTransformer
import datetime
import pandas as pd
import joblib
import scipy.sparse as sp
from sklearn.cluster import KMeans
import time
import traceback
import asyncio
from typing import List, Tuple, Dict
from flask import Flask, request, jsonify
from urllib.parse import unquote


"""**Collect all Reddit data from past week with ZERO filtering**"""
openai_api_key = os.getenv('OPENAI_API_KEY')
nlp = spacy.load("en_core_web_sm")
ENTITY_LABELS_TO_KEEP = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT", "WORK_OF_ART"}

ENTITY_STOPLIST = {
    "Reddit", "YouTube", "Instagram", "Twitter",
    "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday",
    "Today","Yesterday","Tomorrow"
}

def top_spacy_entities(texts: list[str], top_k=None) -> list[str]:
    """
    Extracts all individual Nouns and Proper Nouns from the input texts,
    ignoring common stoplist terms.
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
    Takes the pre-extracted list of nouns and ensures the original 
    query is included, creating the final set of unique search terms.
    """
    terms = set(noun_list)

    if query.strip():
        terms.add(query.strip()) 

    return list(terms)

# =========================
# Embeddings / OpenAI / Reddit
# =========================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = openai.OpenAI(api_key=openai_api_key)

reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent='HFHIve',
    check_for_async=False
)

end_date = datetime.datetime.utcnow().date()
start_date = end_date - datetime.timedelta(days=14)

start_ts = int(datetime.datetime.combine(start_date, datetime.time.min).timestamp())


def fetch_posts(query, subreddit_name="all", limit=None):
    """OPTIMIZED: Fetch more posts with better coverage"""
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


def naive_count_proper_nouns(texts_list):
    """
    OPTIMIZED: Count proper nouns for relevancy ranking
    """
    pattern = re.compile(r'\b[A-Z][a-z]+\b')
    return sum(len(pattern.findall(text)) for text in texts_list)


def get_word_frequencies(texts_list, stopwords=None):
    """Calculate term frequencies in cluster"""
    if stopwords is None:
        stopwords = {"the", "and", "this", "that", "with", "from", "for", "was", "were", "are", "is", "a", "an"}
    all_text = " ".join(texts_list).lower()
    words = re.findall(r'\w+', all_text)
    words = [w for w in words if w not in stopwords]
    return Counter(words)


def get_cluster_keywords(cluster_texts: list[str], top_n: int = 5) -> list[str]:
    """Extract top keywords from a cluster for context"""
    freq = get_word_frequencies(cluster_texts)
    return [word for word, _ in freq.most_common(top_n)]


def format_cluster_for_api(cid: int, cluster_texts: list[str], max_texts: int = 30) -> str:
    """
    Format cluster data for single API call with maximum context.
    Includes keywords, sample texts, and metadata.
    """
    keywords = get_cluster_keywords(cluster_texts, top_n=8)
    proper_noun_count = naive_count_proper_nouns(cluster_texts)
    
    formatted = f"""
CLUSTER {cid}:
- Size: {len(cluster_texts)} posts
- Top Keywords: {", ".join(keywords)}
- Named Entities (proper nouns): {proper_noun_count}
- Sample Posts (up to {min(max_texts, len(cluster_texts))} most relevant):
"""
    
    # Include top posts by length/detail (longer = more info)
    sorted_texts = sorted(cluster_texts, key=len, reverse=True)
    for i, text in enumerate(sorted_texts[:max_texts], 1):
        formatted += f"\n  [{i}] {text[:300]}"  # Truncate to 300 chars per post
    
    return formatted


# =========================
# OPTIMIZED: Single batch OpenAI call with Responses API
# =========================

async def generate_reports_async(query: str, context_question: str, clusters: Dict, proper_counts: dict) -> Tuple[str, str]:
    """
    OPTIMIZED: Single OpenAI call that generates ALL reports at once.
    Reduces from ~5 API calls to 1 comprehensive call.
    Uses new Responses API format.
    """
    
    is_question = bool(context_question)
    
    # Prepare cluster data for API (all clusters in one payload)
    top_clusters = sorted(proper_counts, key=proper_counts.get, reverse=True)[:5]
    
    cluster_data = "\n---\n".join([
        format_cluster_for_api(cid, clusters[cid], max_texts=25)
        for cid in top_clusters
    ])
    
    prompt = f"""You are an expert trend analyst analyzing Reddit conversations. 
Your task is to generate a comprehensive trend report from multiple conversation clusters.

Original Query/Topic: "{query}"
{"Question Focus: " + context_question if is_question else ""}

CLUSTER DATA:
{cluster_data}

---

GENERATE A SINGLE COMPREHENSIVE JSON RESPONSE with this exact structure:
{{
  "headline": "Main trend headline (≤15 words, Title Case, includes entities, includes verb)",
  "executive_summary": "2-3 sentence overview of the dominant narrative",
  "key_insights": [
    "Insight 1: Specific development with named entities (2 sentences max)",
    "Insight 2: Major theme or pattern (2 sentences max)",
    "Insight 3: Controversial or polarizing take mentioned (2 sentences max)"
  ],
  "community_voices": [
    "Direct quote from discussion (≤20 words)",
    "Direct quote showing disagreement (≤20 words)",
    "Direct quote showing majority sentiment (≤20 words)"
  ],
  "future_outlook": "One sentence describing likely next development or shift",
  "off_topic_note": "Only include this if majority of posts are NOT about the query - briefly state what they're actually discussing"
}}

REQUIREMENTS:
1. ALL content must come directly from provided cluster data
2. Include 2-5 specific named entities/locations mentioned in posts
3. Keep neutral, analytical tone (NOT journalistic)
4. If majority posts are off-topic, explain why in off_topic_note
5. Cite actual discussion patterns from the clusters
6. Return ONLY valid JSON, no markdown formatting or extra text
"""

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.responses.create(
                model="gpt-4o",
                input=[
                    {"role": "user", "content": prompt}
                ]
            )
        )
        
        # Extract text from Responses API format
        response_text = response.output_text if hasattr(response, 'output_text') else ""
        
        if not response_text:
            # Fallback: manually extract from output array
            for item in response.output:
                for content in item.get('content', []):
                    if content.get('type') == 'output_text':
                        response_text = content.get('text', '')
                        break
        
        # Parse JSON response
        try:
            import json
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                report_data = json.loads(json_match.group())
            else:
                report_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            report_data = {
                "headline": "Trend Analysis",
                "executive_summary": response_text,
                "key_insights": [],
                "community_voices": [],
                "future_outlook": ""
            }
        
        return report_data, None
        
    except Exception as e:
        error_msg = f"[OpenAI error]: {str(e)}"
        print(error_msg)
        return None, error_msg


def format_final_report(report_data: dict, query: str) -> str:
    """Format the API response into readable markdown"""
    if not report_data:
        return ""
    
    output = f"""# Trend Report: {query}

## {report_data.get('headline', 'Trend Analysis')}

### Executive Summary
{report_data.get('executive_summary', '')}

### Key Insights
"""
    
    for i, insight in enumerate(report_data.get('key_insights', []), 1):
        output += f"\n{i}. {insight}"
    
    output += "\n\n### Community Voices\n"
    for i, quote in enumerate(report_data.get('community_voices', []), 1):
        output += f"\n**Quote {i}:** \"{quote}\"\n"
    
    output += f"\n### Future Outlook\n{report_data.get('future_outlook', '')}\n"
    
    if report_data.get('off_topic_note'):
        output += f"\n⚠️ **Note:** {report_data['off_topic_note']}\n"
    
    return output


# =========================
# Main processing pipeline
# =========================

def summarize_clusters_wrapper(query, context, context_question):
    """OPTIMIZED: Fewer API calls, max data collection"""
    try:
        query = (query or "").strip()
        context = (context or "").strip()
        context_question = (context_question or "").strip()

        if not query:
            return {"error": "Please enter a word or phrase to search."}

        # --- 1) REDDIT DATA FETCHING (INCREASED LIMIT) ---
        all_posts = []
        
        # Primary query
        try:
            print(f"[INFO] Fetching posts for: {query}")
            base_posts = fetch_posts(f"{query}", subreddit_name="all", limit=250)  # INCREASED from 150
            all_posts.extend(base_posts)
            print(f"[INFO] Fetched {len(base_posts)} posts from primary query")
        except Exception as e:
            print(f"[WARN] Reddit fetch failed: {e}")

        # Secondary query with context for more coverage
        if context:
            try:
                print(f"[INFO] Fetching posts for: {query} + {context}")
                context_posts = fetch_posts(f"{query} {context}", subreddit_name="all", limit=200)
                all_posts.extend(context_posts)
                print(f"[INFO] Fetched {len(context_posts)} posts from context query")
            except Exception as e:
                print(f"[WARN] Context fetch failed: {e}")

        if not all_posts:
            return {"error": f"No Reddit posts found for query: '{query}'."}

        posts_df = pd.DataFrame(all_posts)
        posts_df = posts_df.drop_duplicates(subset="post_id")
        print(f"[INFO] Total unique posts after dedup: {len(posts_df)}")

        # Filter to past 7 days (keep most recent data)
        posts_df["time"] = pd.to_datetime(posts_df["created_utc"], unit="s")
        seven_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=7)
        posts_df = posts_df[posts_df["time"] >= seven_days_ago]
        print(f"[INFO] Posts from past 7 days: {len(posts_df)}")

        # OPTIMIZED: Combine title + content, keep all posts (NO FILTERING)
        reddit_texts = (
            posts_df["title"].fillna("") + " " + posts_df["content"].fillna("")
        ).tolist()

        texts = reddit_texts
        if not texts:
            return {"error": f"No social media texts found for query: '{query}'."}

        # REMOVED: Pop culture filtering logic - keep ALL data
        filtered_texts = texts
        
        # OPTIMIZED: Deduplicate exact matches before embedding
        filtered_texts = list(dict.fromkeys(filtered_texts))
        print(f"[INFO] After dedup: {len(filtered_texts)} unique texts")
        
        # Remove very short texts (noise)
        filtered_texts = [t for t in filtered_texts if len(t.strip()) > 20]
        print(f"[INFO] After removing short texts: {len(filtered_texts)} texts")
        
        processed_texts = [simple_preprocess(t) for t in filtered_texts]

        # --- 2) EMBEDDING ---
        print(f"[INFO] Encoding {len(processed_texts)} texts...")
        start_embed = time.time()
        embeddings = embed_model.encode(processed_texts, batch_size=128, show_progress_bar=False)
        print(f"[INFO] Embedding took {time.time() - start_embed:.2f}s")

        # --- 3) CLUSTERING (OPTIMIZED) ---
        n = len(filtered_texts)

        if n < 3:
            clusters = {0: filtered_texts}
            print(f"[INFO] Too few texts ({n}), using single cluster")
        else:
            # OPTIMIZED: Better clustering parameters for relevancy
            min_cluster_size = max(5, min(20, n // 8))  # Larger clusters = more coherent themes
            min_samples = max(1, min_cluster_size // 3)

            print(f"[INFO] Clustering with min_size={min_cluster_size}, min_samples={min_samples}")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                gen_min_span_tree=True
            )

            labels = clusterer.fit_predict(embeddings)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"[INFO] HDBSCAN found {n_clusters} clusters")

            if (labels == -1).all():
                print("[INFO] HDBSCAN failed → fallback to KMeans")
                k = max(3, min(8, n // 20))
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(embeddings)

            clusters = {}
            for text, lbl in zip(filtered_texts, labels):
                if lbl == -1:  # Reassign noise to nearest cluster
                    continue
                clusters.setdefault(lbl, []).append(text)

        print(f"[INFO] Final cluster count: {len(clusters)}")

        # --- 4) RELEVANCY RANKING ---
        proper_counts = {
            cid: naive_count_proper_nouns(txts)
            for cid, txts in clusters.items()
            if cid != -1
        }

        if not proper_counts:
            clusters = {0: filtered_texts}
            proper_counts = {0: naive_count_proper_nouns(filtered_texts)}

        # --- 5) SINGLE BATCH OPENAI CALL ---
        print("[INFO] Generating reports with single API call...")
        start_api = time.time()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            report_data, api_error = loop.run_until_complete(
                generate_reports_async(query, context_question, clusters, proper_counts)
            )
            
            api_time = time.time() - start_api
            print(f"[INFO] API call completed in {api_time:.2f}s")
            
            if api_error:
                return {"error": api_error}
            
            detailed_briefings = format_final_report(report_data, query)
            
        finally:
            loop.close()

        return {
            "query": query,
            "context": context,
            "context_question": context_question,
            "post_count": len(filtered_texts),
            "cluster_count": len(proper_counts),
            "executive_report": report_data.get('executive_summary', '') if report_data else "",
            "detailed_briefings": detailed_briefings,
            "report_data": report_data,  # Include full structured data
        }

    except Exception as e:
        error_msg = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(error_msg)
        return error_msg


# =========================
# Flask API
# =========================
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/hive", methods=["GET", "POST"])
def hive():
    if request.method == "GET":
        query = request.args.get("query", "").strip()
        context = request.args.get("context", "").strip()
        context_question = request.args.get("context_question", "").strip()
    else:
        body = request.get_json(force=True, silent=True) or {}
        query = body.get("query", "").strip()
        context = body.get("context", "").strip()
        context_question = body.get("context_question", "").strip()

    if not query:
        return jsonify({"error": "query field is required"}), 400

    result = summarize_clusters_wrapper(query, context, context_question)
    if "error" in result:
        return jsonify(result), 422
    return jsonify(result), 200

@app.route("/renderhive/<path:query_context>", methods=["GET"])
def renderhive(query_context):
    """
    Handles URLs like /renderhive/query/context where both can contain spaces.
    Properly handles both %20 (URL-encoded spaces) and + (plus-encoded spaces).
    """
    parts = query_context.rsplit('/', 1)
    if len(parts) == 2:
        query, context = parts
    elif len(parts) == 1:
        query = parts[0]
        context = ""
    else:
        return jsonify({"error": "Invalid URL format"}), 400

    query = unquote(query).replace('+', ' ').strip()
    context = (unquote(context).replace('+', ' ').strip()) if context else ""

    result = summarize_clusters_wrapper(query, context, "")
    if "error" in result:
        return jsonify(result), 422
    return jsonify(result), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
