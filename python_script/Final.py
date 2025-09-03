#!/usr/bin/env python3
import os
import re
from collections import Counter, defaultdict
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

DEVELOPER_KEY = "YOUR_KEY_HERE
CHANNEL_ID    = "YOUR_CHANNEL_ID_HERE"
MAX_RESULTS   = 50
MAX_PAGES     = 15

vader = SentimentIntensityAnalyzer()
def sentiment_score(text: str) -> float:
    return vader.polarity_scores(text).get("compound", 0.0)

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

def get_youtube_client(api_key: str):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    return build("youtube", "v3", developerKey=api_key)

def get_videos(channel_id: str, youtube) -> list[str]:
    ids = []
    req = youtube.search().list(part="id", channelId=channel_id, maxResults=MAX_RESULTS, type="video")
    for _ in range(MAX_PAGES):
        resp = req.execute()
        for itm in resp.get("items", []):
            vid = itm["id"].get("videoId")
            if vid:
                ids.append(vid)
        token = resp.get("nextPageToken")
        if token:
            req = youtube.search().list(
                part="id",
                channelId=channel_id,
                maxResults=MAX_RESULTS,
                type="video",
                pageToken=token
            )
        else:
            break
    return ids

def get_video_details(video_ids: list[str], youtube) -> dict:
    info = {}
    for i in range(0, len(video_ids), MAX_RESULTS):
        batch = video_ids[i : i + MAX_RESULTS]
        resp = youtube.videos().list(
            part="snippet,contentDetails",
            id=",".join(batch)
        ).execute()
        for itm in resp.get("items", []):
            vid = itm["id"]
            title = itm["snippet"]["title"]
            cd = itm.get("contentDetails", {})
            cr = cd.get("contentRating", {})
            restricted = cr.get("ytRating") == "ytAgeRestricted"
            info[vid] = {"title": title, "restricted": restricted}
    return info

def get_transcript(video_id: str) -> str:
    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(e["text"] for e in entries)
    except (TranscriptsDisabled, NoTranscriptFound):
        return ""
    except Exception as e:
        print(f"⚠️ Transcript error for {video_id}: {e}")
        return ""

def get_comments(video_id: str, youtube) -> pd.DataFrame:
    rows = []
    req = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=MAX_RESULTS,
        textFormat="plainText",
        order="time"
    )
    while req:
        try:
            resp = req.execute()
        except HttpError as e:
            print(f"⚠️ Comment fetch error for {video_id}: {e}")
            break
        for itm in resp.get("items", []):
            s = itm["snippet"]["topLevelComment"]["snippet"]
            text = s["textDisplay"]
            rows.append({
                "video_id":     video_id,
                "author":       s["authorDisplayName"],
                "comment":      text,
                "likes":        s["likeCount"],
                "published_at": s["publishedAt"],
                "sentiment":    sentiment_score(text)
            })
        req = youtube.commentThreads().list_next(req, resp)
    return pd.DataFrame(rows)

def analyze_channel(channel_id: str, api_key: str):
    yt   = get_youtube_client(api_key)
    vids = get_videos(channel_id, yt)
    details = get_video_details(vids, yt)

    transcripts   = []
    all_comments  = pd.DataFrame()
    word_response = defaultdict(Counter)
    skipped = 0

    for vid in vids:
        info = details.get(vid, {})
        if info.get("restricted"):
            skipped += 1
            continue

        title      = info.get("title", "UNKNOWN")
        transcript = get_transcript(vid)
        t_tokens   = set(tokenize(transcript))

        transcripts.append({
            "video_id":             vid,
            "title":                title,
            "transcript":           transcript,
            "transcript_sentiment": sentiment_score(transcript)
        })

        dfc = get_comments(vid, yt)
        dfc["title"] = title
        all_comments = pd.concat([all_comments, dfc], ignore_index=True)

        for _, row in dfc.iterrows():
            c_tokens = tokenize(row["comment"])
            for t in t_tokens:
                for c in c_tokens:
                    word_response[t][c] += 1

    tf_counter = Counter()
    cf_counter = Counter()
    ts_dict    = defaultdict(list)
    cs_dict    = defaultdict(list)

    for rec in transcripts:
        tf = Counter(tokenize(rec["transcript"]))
        for w, cnt in tf.items():
            tf_counter[w]   += cnt
            ts_dict[w].append(rec["transcript_sentiment"])

    for _, row in all_comments.iterrows():
        cf = Counter(tokenize(row["comment"]))
        for w, cnt in cf.items():
            cf_counter[w]   += cnt
            cs_dict[w].append(row["sentiment"])

    records = []
    for w in set(cf_counter) | set(tf_counter):
        avg_c = sum(cs_dict[w]) / len(cs_dict[w]) if cs_dict[w] else 0
        avg_t = sum(ts_dict[w]) / len(ts_dict[w]) if ts_dict[w] else 0
        records.append({
            "word":                   w,
            "comment_count":          cf_counter.get(w, 0),
            "transcript_count":       tf_counter.get(w, 0),
            "avg_comment_sentiment":  avg_c,
            "avg_transcript_sentiment": avg_t
        })

    df_wordfreq = (
        pd.DataFrame(records)
          .sort_values("comment_count", ascending=False)
          .reset_index(drop=True)
    )

    with pd.ExcelWriter(f"youtube_analysis_{channel_id}.xlsx", engine="xlsxwriter") as writer:
        pd.DataFrame(transcripts).to_excel(writer, sheet_name="Transcripts", index=False)
        all_comments.to_excel(writer, sheet_name="Comments", index=False)
        df_wordfreq.to_excel(writer, sheet_name="WordFreq", index=False)

        sent = all_comments.groupby("video_id")["sentiment"].agg(
            avg_sentiment="mean",
            median_sentiment="median",
            n_comments="count"
        ).reset_index()

        sent = sent.merge(
            pd.DataFrame(transcripts)[["video_id","title","transcript_sentiment"]],
            on="video_id"
        )[
            ["video_id","title","transcript_sentiment","avg_sentiment","median_sentiment","n_comments"]
        ]
        sent.to_excel(writer, sheet_name="SentimentSummary", index=False)

        resp_recs = []
        for t_word, ctr in word_response.items():
            for c_word, cnt in ctr.most_common(5):
                resp_recs.append({
                    "transcript_word": t_word,
                    "response_word":   c_word,
                    "count":           cnt
                })
        pd.DataFrame(resp_recs).to_excel(writer, sheet_name="ResponsePatterns", index=False)

    print(f"✅ Done. Skipped {skipped} age-restricted videos.")
    print(f"🔽 Output: youtube_analysis_{channel_id}.xlsx")

if __name__ == "__main__":
    analyze_channel(CHANNEL_ID, DEVELOPER_KEY)
