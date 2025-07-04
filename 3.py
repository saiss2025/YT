import streamlit as st
import os
import re
import requests
import ssl
from googleapiclient.discovery import build
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime
from langdetect import detect
from googletrans import Translator
import pandas as pd
from gtts import gTTS
from io import BytesIO
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from fpdf import FPDF
import imghdr

# --- Setup ---
ssl._create_default_https_context = ssl.create_default_context
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
ASSET_PDF_PATH, ASSET_LINKS_PATH = "assets/pdfs", "assets/links"
os.makedirs(ASSET_PDF_PATH, exist_ok=True)
os.makedirs(ASSET_LINKS_PATH, exist_ok=True)
translator = Translator()
MAX_INPUT_CHARS = 10000

def truncate(text): return text[:MAX_INPUT_CHARS] if len(
    text) > MAX_INPUT_CHARS else text

def clean_text(text): return re.sub(r'[^\x00-\x7F]+', '', text)

def format_date(iso):
    try:
        return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%d-%m-%Y")
    except:
        return iso.split("T")[0]

def extract_video_id(url):
    m = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", url)
    return m.group(1) if m else None

def extract_channel_id(url):
    if "channel/" in url:
        return url.split("channel/")[-1].split("/")[0]
    if "user/" in url:
        uname = url.split("user/")[-1].split("/")[0]
        res = youtube.channels().list(forUsername=uname, part="id").execute()
        return res["items"][0]["id"] if res["items"] else None
    if "/@" in url:
        handle = url.split("@")[-1].split("/")[0]
        res = youtube.search().list(
            q=f"@{handle}", type="channel", part="snippet", maxResults=1).execute()
        return res["items"][0]["snippet"]["channelId"] if res["items"] else None
    return None

def get_transcript(video_id):
    try:
        data = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([t["text"] for t in data])
        lang = detect(text)
        if lang != 'en':
            text = translator.translate(text, src=lang, dest='en').text
        return text
    except:
        return None

def generate_local_response(prompt):
    try:
        headers = {"Authorization": "Bearer lm-studio"}
        payload = {
            "model": "gemma-3b",
            "messages": [
                {"role": "system", "content": "You are a strict summarizer. Only return the final answer. Do not include any questions, follow-up suggestions, or extra comments. Respond only with the summary or key points as requested. Nothing more."},
                {"role": "user", "content": truncate(prompt)}
            ],
            "temperature": 0.7, "max_tokens": 1024
        }
        r = requests.post(
            "http://localhost:1234/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"LLM Error: {e}"

def get_video_comments(video_id, max_results=20):
    comments = []
    try:
        results = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=max_results, textFormat="plainText"
        ).execute()
        for item in results.get("items", []):
            c = item["snippet"]["topLevelComment"]["snippet"]
            text = c.get("textDisplay", "")
            likes = c.get("likeCount", 0)
            pos_words = ["good", "great", "awesome", "love",
                         "excellent", "amazing", "helpful", "best", "nice", "thank"]
            neg_words = ["bad", "worst", "hate", "poor", "dislike",
                         "boring", "waste", "problem", "issue", "difficult"]
            sentiment = "neutral"
            if any(w in text.lower() for w in pos_words):
                sentiment = "positive"
            elif any(w in text.lower() for w in neg_words):
                sentiment = "negative"
            comments.append(
                {"text": text, "likeCount": likes, "sentiment": sentiment})
    except Exception as e:
        pass
    return comments

def get_video_summary(video_id):
    v = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
    if not v["items"]:
        return None
    s, stats = v["items"][0]["snippet"], v["items"][0]["statistics"]
    title = s.get("title", "No title")
    views = stats.get("viewCount", "N/A")
    likes = stats.get("likeCount", "N/A") if "likeCount" in stats else "N/A"
    transcript = get_transcript(video_id)
    comments = get_video_comments(video_id, max_results=20)
    top_comments = sorted(
        comments, key=lambda x: x["likeCount"], reverse=True)[:3]
    pos_comments = [c for c in comments if c["sentiment"] == "positive"][:3]
    neg_comments = [c for c in comments if c["sentiment"] == "negative"][:3]
    if transcript:
        learnings_prompt = (
            "Give a detailed summary of what we learn and can make (useful) from this YouTube video. "
            "Write a single paragraph of 4 lines. Then, in a new line, list the main topics covered in the video as bullet points."
            "\n\nTranscript:\n" + transcript
        )
        summary_and_topics = generate_local_response(learnings_prompt)
        summary_and_topics = f"<b>Title:</b> {title}<br><br>{summary_and_topics.replace('\n', '<br>')}"
    else:
        summary_and_topics = f"<b>Title:</b> {title}<br><br>Transcript not available."
    return {
        "id": video_id,
        "title": title,
        "views": views,
        "likes": likes,
        "summary_and_topics": summary_and_topics,
        "top_comments": top_comments,
        "pos_comments": pos_comments,
        "neg_comments": neg_comments
    }

def get_channel_details(channel_id):
    ch = youtube.channels().list(
        part="snippet,statistics,brandingSettings,contentDetails", id=channel_id).execute()
    if not ch["items"]:
        return None
    c = ch["items"][0]
    s, stats, b = c.get("snippet", {}), c.get(
        "statistics", {}), c.get("brandingSettings", {})
    uploads = c.get("contentDetails", {}).get(
        "relatedPlaylists", {}).get("uploads", "")
    return {
        "title": s.get("title", "N/A"),
        "pp": s.get("thumbnails", {}).get("high", {}).get("url", ""),
        "banner": b.get("image", {}).get("bannerExternalUrl", ""),
        "intro_video": b.get("channel", {}).get("unsubscribedTrailer", None),
        "uploads": uploads,
        "description": s.get("description", ""),
        "subs": stats.get("subscriberCount", "N/A"),
        "views": stats.get("viewCount", "N/A"),
        "videos": stats.get("videoCount", "N/A"),
        "created": format_date(s.get("publishedAt", "")),
        "country": s.get("country", "N/A"),
        "customUrl": s.get("customUrl", "N/A"),
        "lang": s.get("defaultLanguage", "N/A")
    }

def get_best_video(channel_id):
    ch = youtube.channels().list(part="contentDetails", id=channel_id).execute()
    if not ch["items"]:
        return None
    uploads_playlist = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    vids = []
    nextPageToken = None
    while True:
        pl = youtube.playlistItems().list(
            playlistId=uploads_playlist, part="contentDetails", maxResults=50, pageToken=nextPageToken
        ).execute()
        vids += [item["contentDetails"]["videoId"] for item in pl["items"]]
        nextPageToken = pl.get("nextPageToken")
        if not nextPageToken:
            break
    max_views = -1
    best_video_id = None
    for i in range(0, len(vids), 50):
        batch = vids[i:i+50]
        vdata = youtube.videos().list(part="statistics", id=",".join(batch)).execute()
        for item in vdata["items"]:
            views = int(item["statistics"].get("viewCount", 0))
            if views > max_views:
                max_views = views
                best_video_id = item["id"]
    return best_video_id

def get_top_videos_stats(channel_id, n=5, days=None):
    ch = youtube.channels().list(part="contentDetails", id=channel_id).execute()
    if not ch["items"]:
        return []
    uploads_playlist = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    vids = []
    nextPageToken = None
    while True:
        pl = youtube.playlistItems().list(
            playlistId=uploads_playlist, part="contentDetails", maxResults=50, pageToken=nextPageToken
        ).execute()
        vids += [item["contentDetails"]["videoId"] for item in pl["items"]]
        nextPageToken = pl.get("nextPageToken")
        if not nextPageToken:
            break
    video_stats = []
    for i in range(0, len(vids), 50):
        batch = vids[i:i+50]
        vdata = youtube.videos().list(part="statistics,snippet",
                                      id=",".join(batch)).execute()
        for item in vdata["items"]:
            published = item["snippet"].get("publishedAt", "")
            if days:
                try:
                    pub_date = datetime.strptime(
                        published, "%Y-%m-%dT%H:%M:%S%z")
                    if (datetime.now(pub_date.tzinfo) - pub_date).days > days:
                        continue
                except:
                    pass
            video_stats.append({
                "id": item["id"],
                "title": item["snippet"].get("title", ""),
                "views": int(item["statistics"].get("viewCount", 0)),
                "likes": int(item["statistics"].get("likeCount", 0)) if "likeCount" in item["statistics"] else 0,
                "published": published
            })
    video_stats.sort(key=lambda x: x["views"], reverse=True)
    return video_stats[:n]

def get_all_videos_stats(channel_id):
    ch = youtube.channels().list(part="contentDetails", id=channel_id).execute()
    if not ch["items"]:
        return []
    uploads_playlist = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    vids = []
    nextPageToken = None
    while True:
        pl = youtube.playlistItems().list(
            playlistId=uploads_playlist, part="contentDetails", maxResults=50, pageToken=nextPageToken
        ).execute()
        vids += [item["contentDetails"]["videoId"] for item in pl["items"]]
        nextPageToken = pl.get("nextPageToken")
        if not nextPageToken:
            break
    video_stats = []
    for i in range(0, len(vids), 50):
        batch = vids[i:i+50]
        vdata = youtube.videos().list(part="statistics,snippet",
                                      id=",".join(batch)).execute()
        for item in vdata["items"]:
            published = item["snippet"].get("publishedAt", "")
            video_stats.append({
                "id": item["id"],
                "title": item["snippet"].get("title", ""),
                "views": int(item["statistics"].get("viewCount", 0)),
                "likes": int(item["statistics"].get("likeCount", 0)) if "likeCount" in item["statistics"] else 0,
                "published": published
            })
    return video_stats

def seo_analyze(text):
    keywords = ["tutorial", "learn", "guide", "how",
                "review", "best", "top", "easy", "beginner"]
    emotional = ["amazing", "incredible", "secret",
                 "ultimate", "free", "simple", "fast"]
    score = 0
    found_keywords = [k for k in keywords if k in text.lower()]
    found_emotional = [w for w in emotional if w in text.lower()]
    if 40 < len(text) < 80:
        score += 1
    if found_keywords:
        score += 1
    if found_emotional:
        score += 1
    if text[0].isupper():
        score += 1
    return {
        "length": len(text),
        "keywords": found_keywords,
        "emotional_words": found_emotional,
        "clarity": text[0].isupper(),
        "score": score,
        "max_score": 4
    }

def growth_estimator(channel_id, filter_days=None, year_filter=None):
    stats = get_all_videos_stats(channel_id)
    if not stats or len(stats) < 2:
        return None, None
    df = pd.DataFrame(stats)
    df["published"] = pd.to_datetime(
        df["published"], errors="coerce", utc=True)
    df = df.dropna(subset=["published"])
    if filter_days:
        df = df[df["published"] >= (pd.Timestamp.now(
            tz="UTC") - pd.Timedelta(days=filter_days))]
    if year_filter == "current":
        df = df[df["published"].dt.year == datetime.now().year]
    elif year_filter == "last":
        df = df[df["published"].dt.year == (datetime.now().year - 1)]
    df = df.sort_values("published")
    if len(df) < 2:
        return None, None
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["views"].values
    model = LinearRegression().fit(X, y)
    next_vid = model.predict([[len(df)]])[0]
    growth = (y[-1] - y[0]) / max(1, len(df)-1)
    return df, f"Estimated next video views: {int(next_vid)}\nAverage growth per video: {int(growth)} views"

def tts_playback(text):
    tts = gTTS(text)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    st.audio(mp3_fp.getvalue(), format="audio/mp3")

# --- PDF Generation ---

class PDF(FPDF):
    def header(self):
        pass  # We'll handle headings manually

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 18)
        self.set_text_color(0, 70, 140)
        self.cell(0, 12, label, ln=True, align='C')
        self.ln(2)

    def section_title(self, label):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, label, ln=True, align='L')
        self.ln(1)

    def normal_text(self, text):
        self.set_font('Arial', '', 12)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 8, text)
        self.ln(1)

    def bullet_list(self, items):
        self.set_font('Arial', '', 12)
        self.set_text_color(20, 20, 20)
        for item in items:
            self.cell(5)
            self.multi_cell(0, 8, u"\u2022 " + item)
        self.ln(1)

    def add_image(self, img_path, w=0, h=0):
        # Only add if file exists and is a valid image
        if os.path.exists(img_path) and imghdr.what(img_path) in ["jpeg", "png", "jpg"]:
            self.image(img_path, w=w, h=h)
            self.ln(2)
        # else: skip silently

    def add_link(self, label, url):
        self.set_text_color(0, 0, 255)
        self.set_font('Arial', 'U', 12)
        self.cell(0, 8, label, ln=True, align='L', link=url)
        self.set_text_color(20, 20, 20)
        self.set_font('Arial', '', 12)

def save_growth_graph(df, channel_title):
    plt.figure(figsize=(8, 4))
    plt.plot(df["published"], df["views"], marker='o', color="#1f77b4")
    plt.title("Channel Growth Over Time", fontsize=16, fontweight='bold')
    plt.xlabel("Published Date", fontsize=12)
    plt.ylabel("Views", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    img_path = os.path.join(
        ASSET_PDF_PATH, f"{clean_text(channel_title)}_growth.png")
    plt.savefig(img_path)
    plt.close()
    return img_path

def save_top_videos_bar(df, channel_title):
    plt.figure(figsize=(8, 4))
    plt.bar(df["title"], df["views"], color="#ff7f0e")
    plt.title("Top Videos by Views", fontsize=14, fontweight='bold')
    plt.xlabel("Video Title", fontsize=12)
    plt.ylabel("Views", fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    img_path = os.path.join(
        ASSET_PDF_PATH, f"{clean_text(channel_title)}_topvideos.png")
    plt.savefig(img_path)
    plt.close()
    return img_path

def strip_html_tags(text):
    return re.sub('<[^<]+?>', '', text)

def generate_full_pdf(details, best, intro, ch_summary_and_topics, top_videos_stats, growth_df, growth_text, all_video_summaries, pdf_filename):
    pdf = PDF()
    pdf.add_page()

    # Main Heading
    pdf.chapter_title(f"{details['title']} Channel Report")

    # Channel Info
    pdf.section_title("Channel Info")
    pdf.normal_text(f"Description: {details['description']}")
    pdf.normal_text(f"Subscribers: {details['subs']}")
    pdf.normal_text(f"Total Views: {details['views']}")
    pdf.normal_text(f"Total Videos: {details['videos']}")
    pdf.normal_text(f"Channel Started: {details['created']}")
    pdf.normal_text(f"Country: {details.get('country', 'N/A')}")
    pdf.normal_text(f"Language: {details.get('lang', 'N/A')}")

    # --- Channel Banner and Icon REMOVED ---

    # What you learn
    pdf.section_title("What you learn from this channel")
    pdf.normal_text(ch_summary_and_topics.replace('\n', ' '))

    # Best Video
    pdf.section_title("Best Video")
    pdf.normal_text(
        f"{best['title']} ({best['views']} views, {best['likes']} likes)")
    pdf.add_link("Watch on YouTube", f"https://youtu.be/{best['id']}")
    pdf.normal_text(strip_html_tags(best['summary_and_topics']))
    # Comments for best video
    if best["top_comments"]:
        pdf.section_title("Top 3 Comments")
        pdf.bullet_list([c["text"] for c in best["top_comments"]])
    if best["pos_comments"]:
        pdf.section_title("Top 3 Positive Comments")
        pdf.bullet_list([c["text"] for c in best["pos_comments"]])
    if best["neg_comments"]:
        pdf.section_title("Top 3 Negative Comments")
        pdf.bullet_list([c["text"] for c in best["neg_comments"]])

    # Intro Video
    if intro:
        pdf.section_title("Intro Video")
        pdf.normal_text(
            f"{intro['title']} ({intro['views']} views, {intro['likes']} likes)")
        pdf.add_link("Watch on YouTube", f"https://youtu.be/{intro['id']}")
        pdf.normal_text(strip_html_tags(intro['summary_and_topics']))

    # Top Videos Table and Bar Chart
    pdf.section_title("Top 5 Videos")
    for v in top_videos_stats:
        pdf.normal_text(
            f"{v['title']} ({v['views']} views, {v['likes']} likes)")
        pdf.add_link("Watch on YouTube", f"https://youtu.be/{v['id']}")
    # Save and add bar chart
    top_df = pd.DataFrame(top_videos_stats)
    bar_img = save_top_videos_bar(top_df, details['title'])
    pdf.add_image(bar_img, w=180)

    # All Videos Summaries
    pdf.section_title("All Video Summaries")
    for v in all_video_summaries:
        pdf.normal_text(
            f"{v['title']} ({v['views']} views, {v['likes']} likes)")
        pdf.add_link("Watch on YouTube", f"https://youtu.be/{v['id']}")
        pdf.normal_text(strip_html_tags(v['summary_and_topics']))
        if v["top_comments"]:
            pdf.section_title("Top 3 Comments")
            pdf.bullet_list([c["text"] for c in v["top_comments"]])
        if v["pos_comments"]:
            pdf.section_title("Top 3 Positive Comments")
            pdf.bullet_list([c["text"] for c in v["pos_comments"]])
        if v["neg_comments"]:
            pdf.section_title("Top 3 Negative Comments")
            pdf.bullet_list([c["text"] for c in v["neg_comments"]])

    # Growth Graph
    if growth_df is not None:
        pdf.section_title("Channel Growth Graph")
        growth_img = save_growth_graph(growth_df, details['title'])
        pdf.add_image(growth_img, w=180)
        pdf.normal_text(growth_text)

    # Save DataFrame as CSV in assets/pdfs
    if growth_df is not None:
        csv_path = os.path.join(
            ASSET_PDF_PATH, f"{clean_text(details['title'])}_growth.csv")
        growth_df.to_csv(csv_path, index=False)

    pdf_file_path = os.path.join(ASSET_PDF_PATH, pdf_filename)
    pdf.output(pdf_file_path)
    return pdf_file_path

# --- Streamlit UI ---
st.set_page_config(page_title="YouTube Analyzer", layout="centered")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
if dark_mode:
    st.markdown(
        "<style>body { background-color: #222; color: #eee; } .stButton>button { color: #222; }</style>",
        unsafe_allow_html=True,
    )

st.markdown('<h1 style="text-align:center; font-size:2.7em; font-weight:bold; color:#004488;">YouTube Channel & Video Analyzer</h1>', unsafe_allow_html=True)
url = st.text_input("Paste any YouTube video/channel URL:")
growth_filter = st.selectbox(
    "Growth graph filter",
    [
        "Last 7 days",
        "Last 28 days",
        "Last 90 days",
        "Last 365 days",
        "Lifetime",
        "Current year",
        "Last year"
    ]
)
growth_days = None
growth_year = None
if growth_filter == "Last 7 days":
    growth_days = 7
elif growth_filter == "Last 28 days":
    growth_days = 28
elif growth_filter == "Last 90 days":
    growth_days = 90
elif growth_filter == "Last 365 days":
    growth_days = 365
elif growth_filter == "Current year":
    growth_year = "current"
elif growth_filter == "Last year":
    growth_year = "last"
time_filter = st.selectbox("Analyze top videos from last...", [
                           "All time", "30 days", "90 days"])
days = None if time_filter == "All time" else (
    30 if time_filter == "30 days" else 90)
if not url:
    st.info("Enter a valid YouTube link to begin.")
    st.stop()
vid_id, ch_id = extract_video_id(url), extract_channel_id(url)

all_video_summaries = []

if ch_id:
    details = get_channel_details(ch_id)
    if details:
        if details['pp']:
            st.image(details['pp'], width=120)
        if details['banner']:
            st.image(details['banner'], use_container_width=True)
        st.markdown(
            f'<h2 style="font-size:2em; font-weight:bold; color:#0055aa;">{details["title"]}</h2>', unsafe_allow_html=True)
        st.markdown(
            f'<h3 style="font-size:1.3em; font-weight:bold;">Channel Info</h3>', unsafe_allow_html=True)
        st.markdown(
            f"<b>Description:</b> {details['description']}", unsafe_allow_html=True)
        st.markdown(
            f"<b>Subscribers:</b> {details['subs']}", unsafe_allow_html=True)
        st.markdown(
            f"<b>Total Views:</b> {details['views']}", unsafe_allow_html=True)
        st.markdown(
            f"<b>Total Videos:</b> {details['videos']}", unsafe_allow_html=True)
        st.markdown(
            f"<b>Channel Started:</b> {details['created']}", unsafe_allow_html=True)
        st.markdown(
            f"<b>Country:</b> {details.get('country', 'N/A')}", unsafe_allow_html=True)
        st.markdown(
            f"<b>Language:</b> {details.get('lang', 'N/A')}", unsafe_allow_html=True)
        ch_summary_and_topics = get_channel_summary_and_topics(
            details['description'])
        st.markdown(
            '<h3 style="font-size:1.25em; font-weight:bold;">What you learn from this channel</h3>', unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:1.1em;'>{ch_summary_and_topics.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
        best_id = get_best_video(ch_id)
        best = get_video_summary(best_id)
        st.markdown(
            '<h3 style="font-size:1.25em; font-weight:bold;">Best Video</h3>', unsafe_allow_html=True)
        st.video(f"https://youtu.be/{best_id}")
        st.markdown(
            f"<b>{best['title']}</b> ({best['views']} views, 👍 {best['likes']} likes)", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:1.1em;'>{best['summary_and_topics']}</div>", unsafe_allow_html=True)
        # Show comments for best video
        if best["top_comments"]:
            st.markdown(
                '<span style="font-size: 1.3em; font-weight: bold;">Top 3 Comments:</span>', unsafe_allow_html=True)
            st.markdown("<ul style='font-size:1.15em;'>",
                        unsafe_allow_html=True)
            for c in best["top_comments"]:
                st.markdown(f"<li>{c['text']}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)
        if best["pos_comments"]:
            st.markdown(
                '<span style="font-size: 1.3em; font-weight: bold;">Top 3 Positive Comments:</span>', unsafe_allow_html=True)
            st.markdown("<ul style='font-size:1.15em;'>",
                        unsafe_allow_html=True)
            for c in best["pos_comments"]:
                st.markdown(f"<li>{c['text']}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)
        if best["neg_comments"]:
            st.markdown(
                '<span style="font-size: 1.3em; font-weight: bold;">Top 3 Negative Comments:</span>', unsafe_allow_html=True)
            st.markdown("<ul style='font-size:1.15em;'>",
                        unsafe_allow_html=True)
            for c in best["neg_comments"]:
                st.markdown(f"<li>{c['text']}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)
        intro_id = details['intro_video']
        intro = get_intro_video_summary(intro_id) if intro_id else None
        if intro:
            st.markdown(
                '<h3 style="font-size:1.25em; font-weight:bold;">Intro Video</h3>', unsafe_allow_html=True)
            st.video(f"https://youtu.be/{intro_id}")
            st.markdown(
                f"<div style='font-size:1.1em;'>{intro['summary_and_topics']}</div>", unsafe_allow_html=True)
        # Top videos, export
        st.markdown(
            '<h3 style="font-size:1.25em; font-weight:bold;">Top 5 Videos</h3>', unsafe_allow_html=True)
        top_videos_stats = get_top_videos_stats(ch_id, 5, days=days)
        for i, vstat in enumerate(top_videos_stats, 1):
            v = get_video_summary(vstat['id'])
            all_video_summaries.append(v)
            st.markdown(
                f"<b>{i}. <a href='https://youtu.be/{v['id']}' style='color:#0055ff;' target='_blank'>{v['title']}</a></b> - {v['views']} views, 👍 {v['likes']} likes", unsafe_allow_html=True)
            with st.expander("Show summary", expanded=False):
                st.markdown(
                    f"<div style='font-size:1.1em;'>{v['summary_and_topics']}</div>", unsafe_allow_html=True)
                if v["top_comments"]:
                    st.markdown(
                        '<span style="font-size: 1.15em; font-weight: bold;">Top 3 Comments:</span>', unsafe_allow_html=True)
                    st.markdown("<ul style='font-size:1.05em;'>",
                                unsafe_allow_html=True)
                    for c in v["top_comments"]:
                        st.markdown(f"<li>{c['text']}</li>",
                                    unsafe_allow_html=True)
                    st.markdown("</ul>", unsafe_allow_html=True)
                if v["pos_comments"]:
                    st.markdown(
                        '<span style="font-size: 1.15em; font-weight: bold;">Top 3 Positive Comments:</span>', unsafe_allow_html=True)
                    st.markdown("<ul style='font-size:1.05em;'>",
                                unsafe_allow_html=True)
                    for c in v["pos_comments"]:
                        st.markdown(f"<li>{c['text']}</li>",
                                    unsafe_allow_html=True)
                    st.markdown("</ul>", unsafe_allow_html=True)
                if v["neg_comments"]:
                    st.markdown(
                        '<span style="font-size: 1.15em; font-weight: bold;">Top 3 Negative Comments:</span>', unsafe_allow_html=True)
                    st.markdown("<ul style='font-size:1.05em;'>",
                                unsafe_allow_html=True)
                    for c in v["neg_comments"]:
                        st.markdown(f"<li>{c['text']}</li>",
                                    unsafe_allow_html=True)
                    st.markdown("</ul>", unsafe_allow_html=True)
                if st.button("🔊 Voice Summary", key=f"voice_{v['id']}"):
                    tts_playback(v['summary_and_topics'])
        # Dashboard
        st.markdown(
            '<h3 style="font-size:1.25em; font-weight:bold;">📊 Channel Growth Graph</h3>', unsafe_allow_html=True)
        growth_df, growth_text = growth_estimator(
            ch_id, filter_days=growth_days, year_filter=growth_year)
        if growth_df is not None:
            st.line_chart(growth_df.set_index("published")["views"])
            st.markdown(
                f"<div style='font-size:1.1em;'>{growth_text}</div>", unsafe_allow_html=True)
        else:
            st.info(
                "Not enough data for growth estimation or no videos in selected period.")
        # SEO Analyzer
        st.markdown(
            '<h3 style="font-size:1.25em; font-weight:bold;">SEO Analyzer</h3>', unsafe_allow_html=True)
        for vstat in top_videos_stats:
            seo = seo_analyze(vstat["title"])
            st.markdown(
                f"<b>{vstat['title']}</b> - SEO Score: {seo['score']}/{seo['max_score']}", unsafe_allow_html=True)
        # PDF Generation at the end
        st.markdown("---")
        st.markdown(
            '<h2 style="font-size:1.5em; font-weight:bold; color:#004488;">Generate Full PDF Report</h2>', unsafe_allow_html=True)
        if st.button("Generate PDF Report"):
            pdf_filename = f"{clean_text(details['title']).replace(' ', '_')}_full_report.pdf"
            pdf_file = generate_full_pdf(
                details, best, intro, ch_summary_and_topics, top_videos_stats, growth_df, growth_text, all_video_summaries, pdf_filename
            )
            with open(pdf_file, "rb") as f:
                st.download_button("⬇️ Download PDF", f,
                                   file_name=os.path.basename(pdf_file))
elif vid_id and not ch_id:
    st.subheader("🎥 Single Video Summary")
    video = get_video_summary(vid_id)
    st.markdown(
        f"<b>Title:</b> {video['title']} ({video['views']} views, 👍 {video['likes']} likes)", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:1.1em;'>{video['summary_and_topics']}</div>", unsafe_allow_html=True)
    if video["top_comments"]:
        st.markdown(
            '<span style="font-size: 1.15em; font-weight: bold;">Top 3 Comments:</span>', unsafe_allow_html=True)
        st.markdown("<ul style='font-size:1.05em;'>", unsafe_allow_html=True)
        for c in video["top_comments"]:
            st.markdown(f"<li>{c['text']}</li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)
    if video["pos_comments"]:
        st.markdown(
            '<span style="font-size: 1.15em; font-weight: bold;">Top 3 Positive Comments:</span>', unsafe_allow_html=True)
        st.markdown("<ul style='font-size:1.05em;'>", unsafe_allow_html=True)
        for c in video["pos_comments"]:
            st.markdown(f"<li>{c['text']}</li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)
    if video["neg_comments"]:
        st.markdown(
            '<span style="font-size: 1.15em; font-weight: bold;">Top 3 Negative Comments:</span>', unsafe_allow_html=True)
        st.markdown("<ul style='font-size:1.05em;'>", unsafe_allow_html=True)
        for c in video["neg_comments"]:
            st.markdown(f"<li>{c['text']}</li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)
    if st.button("🔊 Voice Summary"):
        tts_playback(video['summary_and_topics'])
else:
    st.warning("❌ Could not identify the channel from the URL.")