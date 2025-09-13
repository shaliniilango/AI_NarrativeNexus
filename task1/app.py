import os
import time
import math
from datetime import datetime, timezone
import pandas as pd
import praw
import tweepy
from dotenv import load_dotenv

# ------------------ CONFIG ------------------
load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# desired counts
REDDIT_TARGET = 2500
TWITTER_TARGET = 900

# where to save
REDDIT_CSV = "reddit_posts_2500.csv"
TWITTER_CSV = "twitter_posts_2500.csv"
COMBINED_CSV = "combined_posts.csv"

# choose subreddits and twitter queries
SUBREDDITS = [
    "MachineLearning","datascience","artificial","technology","Python",
    "AI","programming","bigdata","computervision","deeplearning"
]
TWITTER_QUERIES = [
    "artificial intelligence", "data science", "machine learning"
]

# polite pauses (seconds)
REDDIT_SLEEP = 1.5
TWITTER_SLEEP = 1.0

# ------------------ HELPERS ------------------
def validate_env():
    missing = []
    if not REDDIT_CLIENT_ID: missing.append("REDDIT_CLIENT_ID")
    if not REDDIT_CLIENT_SECRET: missing.append("REDDIT_CLIENT_SECRET")
    if not REDDIT_USER_AGENT: missing.append("REDDIT_USER_AGENT")
    if not TWITTER_BEARER_TOKEN: missing.append("TWITTER_BEARER_TOKEN")
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}. Put them in a .env file.")

# ------------------ REDDIT FETCH ------------------
def fetch_reddit_posts(target=REDDIT_TARGET, subreddits=SUBREDDITS, filename=REDDIT_CSV):
    """
    Fetch approximately `target` posts across the provided subreddits.
    Uses subreddit.new() iterator (safer for broad coverage).
    """
    print("Initializing Reddit client...")
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        check_for_async=False
    )

    posts = []
    per_sub_target = math.ceil(target / max(1, len(subreddits)))
    print(f"Target total: {target}. Per-subreddit approx: {per_sub_target}")

    for sub in subreddits:
        print(f"Fetching up to {per_sub_target} posts from r/{sub} ...")
        count = 0
        try:
            # Using .new avoids stuck hot/top internal limits and gives fresh posts
            for submission in reddit.subreddit(sub).new(limit=None):
                posts.append({
                    "id": f"r_{submission.id}",
                    "source": "reddit",
                    "author": submission.author.name if submission.author else "unknown",
                    "timestamp": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat(),
                    "text": (submission.title or "") + ("\n" + (submission.selftext or "") if (submission.selftext) else ""),
                    "subreddit": sub,
                    "likes": submission.score,
                    "url": f"https://www.reddit.com{submission.permalink}"
                })
                count += 1
                if count >= per_sub_target:
                    break
            print(f"  collected {count} from r/{sub}")
        except Exception as e:
            print(f"  Error fetching from r/{sub}: {e}")
        time.sleep(REDDIT_SLEEP)

        # stop early if we reached target
        if len(posts) >= target:
            print("Reached reddit target total.")
            break

    # final trimming to exact target (if overshot)
    if len(posts) > target:
        posts = posts[:target]

    df = pd.DataFrame(posts)
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Saved {len(df)} reddit posts to {filename}")
    return df

# ------------------ TWITTER FETCH ------------------
def fetch_twitter_tweets(target=TWITTER_TARGET, queries=TWITTER_QUERIES, filename=TWITTER_CSV):
    """
    Fetch approximately `target` tweets across the provided queries using tweepy.Client.search_recent_tweets.
    Note: search_recent_tweets returns tweets from the past ~7 days.
    """
    print("Initializing Twitter client...")
    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)

    tweets = []
    per_query_target = math.ceil(target / max(1, len(queries)))
    print(f"Target total: {target}. Per-query approx: {per_query_target}")

    for q in queries:
        print(f"Fetching up to {per_query_target} tweets for query: '{q}' ...")
        count = 0
        # build a query that excludes retweets for cleaner dataset
        query_string = f'{q} -is:retweet lang:en'
        try:
            paginator = tweepy.Paginator(
                client.search_recent_tweets,
                query=query_string,
                tweet_fields=["created_at", "lang", "public_metrics", "author_id", "text"],
                max_results=100
            )
            for page in paginator:
                if page.data is None:
                    continue
                for t in page.data:
                    tweets.append({
                        "id": f"t_{t.id}",
                        "source": "twitter",
                        "author": str(t.author_id),
                        "timestamp": t.created_at.isoformat() if t.created_at else None,
                        "text": t.text,
                        "query": q,
                        "language": t.lang,
                        "likes": t.public_metrics.get("like_count") if t.public_metrics else None,
                        "retweets": t.public_metrics.get("retweet_count") if t.public_metrics else None
                    })
                    count += 1
                    if count >= per_query_target:
                        break
                if count >= per_query_target:
                    break
                # small sleep per page to respect rate limiting
                time.sleep(TWITTER_SLEEP)
            print(f"  collected {count} tweets for query '{q}'")
        except Exception as e:
            print(f"  Error fetching tweets for '{q}': {e}")

        # stop early if reach target overall
        if len(tweets) >= target:
            print("Reached twitter target total.")
            break

    # trim overshoot
    if len(tweets) > target:
        tweets = tweets[:target]

    df = pd.DataFrame(tweets)
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Saved {len(df)} tweets to {filename}")
    return df

# ------------------ COMBINE ------------------
def combine_and_save(reddit_df, twitter_df, filename=COMBINED_CSV):
    print("Combining datasets...")
    # Ensure consistent columns
    reddit_df = reddit_df.copy()
    twitter_df = twitter_df.copy()

    # Some columns may only exist in one of them; unify important ones
    reddit_df["platform_id"] = reddit_df["id"]
    twitter_df["platform_id"] = twitter_df["id"]

    # Create common columns: id, source, author, timestamp, text, metadata-like fields
    common_cols = ["platform_id", "source", "author", "timestamp", "text"]
    # keep all other columns too
    combined = pd.concat([reddit_df, twitter_df], ignore_index=True, sort=False)

    # Drop duplicates by platform_id (safe because ids are prefixed with r_ or t_)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["platform_id"])
    after = len(combined)
    print(f"Dropped {before - after} duplicate rows.")

    # Reset index and save
    combined = combined.reset_index(drop=True)
    combined.to_csv(filename, index=False, encoding="utf-8")
    print(f"Combined dataset saved to {filename} with {len(combined)} rows.")
    return combined

# ------------------ MAIN ------------------
if __name__ == "__main__":
    try:
        validate_env()
    except RuntimeError as e:
        print("ENV ERROR:", e)
        print("Make sure you created a .env file with your credentials.")
        raise SystemExit(1)

    # 1) Reddit
    reddit_df = fetch_reddit_posts(target=REDDIT_TARGET, subreddits=SUBREDDITS, filename=REDDIT_CSV)

    # 2) Twitter
    twitter_df = fetch_twitter_tweets(target=TWITTER_TARGET, queries=TWITTER_QUERIES, filename=TWITTER_CSV)

    # 3) Combine
    combined_df = combine_and_save(reddit_df, twitter_df, filename=COMBINED_CSV)

    print("All done.")
