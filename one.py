import streamlit as st, os, re, requests, ssl
from googleapiclient.discovery import build
from fpdf import FPDF
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

# --- Setup ---
ssl._create_default_https_context = ssl.create_default_context  # Uncomment if SSL errors
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
ASSET_PDF_PATH, ASSET_LINKS_PATH = "assets/pdfs", "assets/links"
os.makedirs(ASSET_PDF_PATH, exist_ok=True)
os.makedirs(ASSET_LINKS_PATH, exist_ok=True)
translator = Translator()
MAX_INPUT_CHARS = 10000

# --- Utils ---
def truncate(text): return text[:MAX_INPUT_CHARS] if len(text) > MAX_INPUT_CHARS else text
def clean_text(text): return re.sub(r'[^\x00-\x7F]+', '', text)
def format_date(iso): 
    try: return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%d-%m-%Y")
    except: return iso.split("T")[0]
def extract_video_id(url): 
    m = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", url)
    return m.group(1) if m else None
def extract_channel_id(url):
    if "channel/" in url: return url.split("channel/")[-1].split("/")[0]
    if "user/" in url:
        uname = url.split("user/")[-1].split("/")[0]
        res = youtube.channels().list(forUsername=uname, part="id").execute()
        return res["items"][0]["id"] if res["items"] else None
    if "/@" in url:
        handle = url.split("@")[-1].split("/")[0]
        res = youtube.search().list(q=f"@{handle}", type="channel", part="snippet", maxResults=1).execute()
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
    except: return None

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
        r = requests.post("http://localhost:1234/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e: return f"LLM Error: {e}"

def get_video_comments(video_id, max_results=20):
    # Returns a list of dicts: {text, likeCount, sentiment}
    comments = []
    try:
        results = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=max_results, textFormat="plainText"
        ).execute()
        for item in results.get("items", []):
            c = item["snippet"]["topLevelComment"]["snippet"]
            text = c.get("textDisplay", "")
            likes = c.get("likeCount", 0)
            # Use a simple sentiment check (positive/negative) using keywords
            pos_words = ["good", "great", "awesome", "love", "excellent", "amazing", "helpful", "best", "nice", "thank"]
            neg_words = ["bad", "worst", "hate", "poor", "dislike", "boring", "waste", "problem", "issue", "difficult"]
            sentiment = "neutral"
            if any(w in text.lower() for w in pos_words): sentiment = "positive"
            elif any(w in text.lower() for w in neg_words): sentiment = "negative"
            comments.append({"text": text, "likeCount": likes, "sentiment": sentiment})
    except Exception as e:
        pass
    return comments

def get_video_summary(video_id):
    v = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
    if not v["items"]: return None
    s, stats = v["items"][0]["snippet"], v["items"][0]["statistics"]
    title = s.get("title", "No title")
    views = stats.get("viewCount", "N/A")
    likes = stats.get("likeCount", "N/A") if "likeCount" in stats else "N/A"
    transcript = get_transcript(video_id)
    comments = get_video_comments(video_id, max_results=20)
    # Top 3 comments by likes
    top_comments = sorted(comments, key=lambda x: x["likeCount"], reverse=True)[:3]
    # Top 3 positive and 3 negative comments
    pos_comments = [c for c in comments if c["sentiment"] == "positive"][:3]
    neg_comments = [c for c in comments if c["sentiment"] == "negative"][:3]
    if transcript:
        learnings_prompt = (
            "Give a detailed summary of what we learn and can make (useful) from this YouTube video. "
            "Write a single paragraph of 4 lines. Then, in a new line, list the main topics covered in the video as bullet points."
            "\n\nTranscript:\n" + transcript
        )
        summary_and_topics = generate_local_response(learnings_prompt)
        summary_and_topics = f"**Title:** {title}\n\n{summary_and_topics}"
    else:
        summary_and_topics = f"**Title:** {title}\n\nTranscript not available."
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
    ch = youtube.channels().list(part="snippet,statistics,brandingSettings,contentDetails", id=channel_id).execute()
    if not ch["items"]: return None
    c = ch["items"][0]
    s, stats, b = c.get("snippet", {}), c.get("statistics", {}), c.get("brandingSettings", {})
    uploads = c.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads", "")
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
        vdata = youtube.videos().list(part="statistics,snippet", id=",".join(batch)).execute()
        for item in vdata["items"]:
            published = item["snippet"].get("publishedAt", "")
            if days:
                try:
                    pub_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%S%z")
                    if (datetime.now(pub_date.tzinfo) - pub_date).days > days:
                        continue
                except: pass
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
        vdata = youtube.videos().list(part="statistics,snippet", id=",".join(batch)).execute()
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

def save_pdf(text, filename):
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
    for line in text.split('\n'): pdf.multi_cell(0, 10, line)
    path = os.path.join(ASSET_PDF_PATH, filename)
    pdf.output(path)
    return path

def generate_pdf(details, best, intro, ch_summary_and_topics, filename=None):
    def safe(t): return clean_text(t).encode('latin-1', 'ignore').decode('latin-1')
    pdf = FPDF(); pdf.add_page(); pdf.set_text_color(30, 30, 30)
    pdf.set_font("Arial", 'B', 18); pdf.cell(0, 15, safe(f"{details['title']} Channel Report"), ln=True, align='C')
    pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, "Channel Info:", ln=True, align='L')
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, safe(f"Description: {details['description']}"))
    pdf.multi_cell(0, 8, safe(f"Subscribers: {details['subs']}"))
    pdf.multi_cell(0, 8, safe(f"Total Views: {details['views']}"))
    pdf.multi_cell(0, 8, safe(f"Total Videos: {details['videos']}"))
    pdf.multi_cell(0, 8, safe(f"Channel Started: {details['created']}"))
    pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, "What you learn from this channel:", ln=True, align='L')
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, safe(ch_summary_and_topics))
    pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, "Best Video:", ln=True, align='L')
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, safe(f"{best['title']} ({best['views']} views)"))
    pdf.multi_cell(0, 8, f"https://youtu.be/{best['id']}")
    pdf.multi_cell(0, 8, safe(best['summary_and_topics']))
    if intro:
        pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, "Intro Video:", ln=True, align='L')
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 8, f"https://youtu.be/{intro['id']}")
        pdf.multi_cell(0, 8, safe(intro['summary_and_topics']))
    if not filename:
        filename = f"{clean_text(details['title']).replace(' ', '_')}.pdf"
    pdf_file_path = os.path.join(ASSET_PDF_PATH, filename)
    pdf.output(pdf_file_path)
    return pdf_file_path

def get_channel_summary_and_topics(description):
    prompt = (
        "Give a detailed summary of what we learn from this YouTube channel. "
        "Write a single paragraph of 4 lines. Then, in a new line, list the main topics and learnings provided by this channel as bullet points."
        "\n\nChannel Description:\n" + description
    )
    return generate_local_response(prompt)

def get_intro_video_summary(intro_id):
    if not intro_id:
        return None
    return get_video_summary(intro_id)

def seo_analyze(text):
    # Simple SEO checks: length, keywords, emotional words, clarity
    keywords = ["tutorial", "learn", "guide", "how", "review", "best", "top", "easy", "beginner"]
    emotional = ["amazing", "incredible", "secret", "ultimate", "free", "simple", "fast"]
    score = 0
    found_keywords = [k for k in keywords if k in text.lower()]
    found_emotional = [w for w in emotional if w in text.lower()]
    if 40 < len(text) < 80: score += 1
    if found_keywords: score += 1
    if found_emotional: score += 1
    if text[0].isupper(): score += 1
    return {
        "length": len(text),
        "keywords": found_keywords,
        "emotional_words": found_emotional,
        "clarity": text[0].isupper(),
        "score": score,
        "max_score": 4
    }

def growth_estimator(channel_id, filter_days=None, year_filter=None):
    # Use all videos for growth graph, filter by days or year if provided
    stats = get_all_videos_stats(channel_id)
    if not stats or len(stats) < 2:
        return None, None
    df = pd.DataFrame(stats)
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)  # Ensure UTC
    df = df.dropna(subset=["published"])
    if filter_days:
        df = df[df["published"] >= (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=filter_days))]
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

# --- Streamlit UI ---
st.set_page_config(page_title="YouTube Analyzer", layout="centered")
# Dark mode toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
if dark_mode:
    st.markdown(
        "<style>body { background-color: #222; color: #eee; } .stButton>button { color: #222; }</style>",
        unsafe_allow_html=True,
    )
st.title("YouTube Channel & Video Analyzer")
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
# For top videos filter
time_filter = st.selectbox("Analyze top videos from last...", ["All time", "30 days", "90 days"])
days = None if time_filter == "All time" else (30 if time_filter == "30 days" else 90)
if not url: st.info("Enter a valid YouTube link to begin."); st.stop()
vid_id, ch_id = extract_video_id(url), extract_channel_id(url)

if ch_id:
    details = get_channel_details(ch_id)
    if details:
        if details['pp']: st.image(details['pp'], width=100)
        if details['banner']: st.image(details['banner'], use_container_width=True)
        st.markdown(f"### {details['title']}")
        st.write(f"**Description:** {details['description']}")
        st.write(f"**Subscribers:** {details['subs']}")
        st.write(f"**Views:** {details['views']}")
        st.write(f"**Total Videos:** {details['videos']}")
        st.write(f"**Channel Started:** {details['created']}")
        st.write(f"**Country:** {details.get('country','N/A')}")
        st.write(f"**Language:** {details.get('lang','N/A')}")
        ch_summary_and_topics = get_channel_summary_and_topics(details['description'])
        st.markdown("### What you learn from this channel")
        st.info(ch_summary_and_topics)
        best_id = get_best_video(ch_id)
        best = get_video_summary(best_id)
        st.markdown("### Best Video")
        st.video(f"https://youtu.be/{best_id}")
        st.write(f"**{best['title']}** ({best['views']} views, 👍 {best['likes']} likes)")
        st.info(best['summary_and_topics'])
        # Show comments for best video
        if best["top_comments"]:
            st.markdown("**Top 3 Comments:**")
            for c in best["top_comments"]:
                st.write(f"👍 {c['likeCount']} - {c['text']}")
        if best["pos_comments"]:
            st.markdown("**Top 3 Positive Comments:**")
            for c in best["pos_comments"]:
                st.write(f"👍 {c['likeCount']} - {c['text']}")
        if best["neg_comments"]:
            st.markdown("**Top 3 Negative Comments:**")
            for c in best["neg_comments"]:
                st.write(f"👍 {c['likeCount']} - {c['text']}")
        intro_id = details['intro_video']
        intro = get_intro_video_summary(intro_id) if intro_id else None
        if intro:
            st.markdown("### Intro Video")
            st.video(f"https://youtu.be/{intro_id}")
            st.info(intro['summary_and_topics'])
        # Top videos, export
        st.markdown("### Top 5 Videos")
        top_videos_stats = get_top_videos_stats(ch_id, 5, days=days)
        for i, vstat in enumerate(top_videos_stats, 1):
            v = get_video_summary(vstat['id'])
            st.write(f"{i}. [{v['title']}](https://youtu.be/{v['id']}) - {v['views']} views, 👍 {v['likes']} likes")
            with st.expander("Show summary", expanded=False):
                st.info(v['summary_and_topics'])
                if v["top_comments"]:
                    st.markdown("**Top 3 Comments:**")
                    for c in v["top_comments"]:
                        st.write(f"👍 {c['likeCount']} - {c['text']}")
                if v["pos_comments"]:
                    st.markdown("**Top 3 Positive Comments:**")
                    for c in v["pos_comments"]:
                        st.write(f"👍 {c['likeCount']} - {c['text']}")
                if v["neg_comments"]:
                    st.markdown("**Top 3 Negative Comments:**")
                    for c in v["neg_comments"]:
                        st.write(f"👍 {c['likeCount']} - {c['text']}")
                if st.button("🔊 Voice Summary", key=f"voice_{v['id']}"):
                    tts_playback(v['summary_and_topics'])
        # Dashboard
        st.markdown("### 📊 Channel Growth Graph")
        growth_df, growth_text = growth_estimator(ch_id, filter_days=growth_days, year_filter=growth_year)
        if growth_df is not None:
            st.line_chart(growth_df.set_index("published")["views"])
            st.write(growth_text)
        else:
            st.info("Not enough data for growth estimation or no videos in selected period.")
        # SEO Analyzer
        st.markdown("### SEO Analyzer")
        for vstat in top_videos_stats:
            seo = seo_analyze(vstat["title"])
            st.write(f"**{vstat['title']}** - SEO Score: {seo['score']}/{seo['max_score']}")
        if st.button("📄 Download PDF Report"):
            pdf_file = generate_pdf(details, best, intro, ch_summary_and_topics, filename=f"{clean_text(details['title']).replace(' ','_')}.pdf")
            with open(pdf_file, "rb") as f:
                st.download_button("⬇️ Download PDF", f, file_name=os.path.basename(pdf_file))
        # AI Chatbot (simple retrieval)
        st.markdown("### 🤖 Channel Q&A Chatbot")
        if "chat_history" not in st.session_state: st.session_state.chat_history = []
        user_q = st.text_input("Ask about this channel or its videos:")
        if user_q:
            # Simple retrieval: search summaries
            all_summaries = [get_video_summary(v["id"])["summary_and_topics"] for v in top_videos_stats]
            all_summaries.append(ch_summary_and_topics)
            answer = generate_local_response(f"Answer this user question based on the following summaries:\n{all_summaries}\n\nQ: {user_q}")
            st.session_state.chat_history.append(("user", user_q))
            st.session_state.chat_history.append(("bot", answer))
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Bot:** {msg}")
elif vid_id and not ch_id:
    st.subheader("🎥 Single Video Summary")
    video = get_video_summary(vid_id)
    st.write(f"**Title:** {video['title']} ({video['views']} views, 👍 {video['likes']} likes)")
    st.info(video['summary_and_topics'])
    if video["top_comments"]:
        st.markdown("**Top 3 Comments:**")
        for c in video["top_comments"]:
            st.write(f"👍 {c['likeCount']} - {c['text']}")
    if video["pos_comments"]:
        st.markdown("**Top 3 Positive Comments:**")
        for c in video["pos_comments"]:
            st.write(f"👍 {c['likeCount']} - {c['text']}")
    if video["neg_comments"]:
        st.markdown("**Top 3 Negative Comments:**")
        for c in video["neg_comments"]:
            st.write(f"👍 {c['likeCount']} - {c['text']}")
    if st.button("🔊 Voice Summary"):
        tts_playback(video['summary_and_topics'])
    if st.button("💾 Download Video Summary as PDF"):
        pdf_path = save_pdf(
            f"{video['title']}\n\nWhat you learn from this video:\n{video['summary_and_topics']}",
            f"{clean_text(video['title']).replace(' ','_')}_summary.pdf"
        )
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name=os.path.basename(pdf_path))
else:
    st.warning("❌ Could not identify the channel from the URL.")