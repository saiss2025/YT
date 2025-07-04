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
import matplotlib.pyplot as plt
import imghdr

# --- Setup ---
ssl._create_default_https_context = ssl.create_default_context  # Uncomment if SSL errors
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
ASSET_PDF_PATH = "assets/pdfs"
os.makedirs(ASSET_PDF_PATH, exist_ok=True)
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
    try:
        if "channel/" in url:
            return url.split("channel/")[-1].split("/")[0]
        if "user/" in url:
            uname = url.split("user/")[-1].split("/")[0]
            res = youtube.channels().list(forUsername=uname, part="id").execute()
            return res["items"][0]["id"] if res["items"] else None
        if "/@" in url:
            handle = url.split("@")[-1].split("/")[0]
            try:
                res = youtube.search().list(q=f"@{handle}", type="channel", part="snippet", maxResults=1).execute()
                return res["items"][0]["snippet"]["channelId"] if res["items"] else None
            except Exception as e:
                st.error("YouTube API quota exceeded or error occurred. Please try again later.")
                return None
        return None
    except Exception as e:
        st.error("YouTube API quota exceeded or error occurred. Please try again later.")
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

# ...existing code...

def get_video_comments(video_id, max_results=20):
    comments = []
    try:
        results = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=3, textFormat="plainText", order="relevance"
        ).execute()
        for item in results.get("items", []):
            c = item["snippet"]["topLevelComment"]["snippet"]
            text = c.get("textDisplay", "")
            likes = c.get("likeCount", 0)
            pos_words = ["good", "great", "awesome", "love", "excellent", "amazing", "helpful", "best", "nice", "thank"]
            neg_words = ["bad", "worst", "hate", "poor", "dislike", "boring", "waste", "problem", "issue", "difficult", "improve", "suggest", "should", "could", "missing", "not", "need", "better"]
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
    comments = get_video_comments(video_id, max_results=3)  # Only fetch 3 for speed
    top_comments = comments[:3]
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

def download_image(url, path):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            # Check if it's a valid image
            if imghdr.what(path) in ["jpeg", "png"]:
                return path
            else:
                os.remove(path)
                return None
    except:
        pass
    return None


def plot_growth_graph(df, img_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 3))
    plt.plot(df["published"], df["views"], marker="o", color="#1976d2")
    plt.title("Channel Growth Over Time")
    plt.xlabel("Published Date")
    plt.ylabel("Views")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    return img_path

def save_df_to_csv(df, filename):
    path = os.path.join(ASSET_PDF_PATH, filename)
    df.to_csv(path, index=False)
    return path

class ColorfulPDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 22)
        self.set_text_color(30, 30, 120)
        self.cell(0, 15, self.title, ln=True, align='C')
        self.ln(2)

    def section_title(self, txt):
        self.set_font("Arial", 'B', 16)
        self.set_text_color(40, 40, 40)
        self.cell(0, 10, txt, ln=True, align='L')
        self.ln(1)

    def section_body(self, txt):
        self.set_font("Arial", '', 12)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 8, txt)
        self.ln(1)

    def add_image(self, img_path, w=180):
        if img_path and os.path.exists(img_path) and imghdr.what(img_path) in ["jpeg", "png"]:
            self.image(img_path, w=w)
            self.ln(2)

    def add_link(self, url, text):
        self.set_text_color(21, 101, 192)
        self.set_font("Arial", 'U', 12)
        self.cell(0, 10, text, ln=True, align='L', link=url)
        self.set_text_color(20, 20, 20)
        self.set_font("Arial", '', 12)

def generate_final_pdf(details, ch_summary_and_topics, best, intro, top_videos, growth_img_path, banner_path, pp_path, filename, growth_df):
    pdf = ColorfulPDF()
    pdf.title = f"{details['title']} Channel Report"
    pdf.add_page()

    # Banner
    if banner_path and os.path.exists(banner_path):
        pdf.add_image(banner_path, w=190)
    # Profile Pic
    if pp_path and os.path.exists(pp_path):
        pdf.add_image(pp_path, w=40)


    # Channel Info
    pdf.section_title("Channel Info")
    pdf.section_body(
        f"Description: {details['description']}\n"
        f"Subscribers: {details['subs']}\n"
        f"Total Views: {details['views']}\n"
        f"Total Videos: {details['videos']}\n"
        f"Channel Started: {details['created']}\n"
        f"Country: {details.get('country','N/A')}\n"
        f"Language: {details.get('lang','N/A')}"
    )

    # What you learn
    pdf.section_title("What you learn from this channel")
    pdf.section_body(ch_summary_and_topics)

    # Best Video
    pdf.section_title("Best Video")
    pdf.section_body(f"{best['title']} ({best['views']} views, 👍 {best['likes']} likes)")
    pdf.add_link(f"https://youtu.be/{best['id']}", "Watch on YouTube")
    pdf.section_body(best['summary_and_topics'])

    # Intro Video
    if intro:
        pdf.section_title("Intro Video")
        pdf.section_body(f"{intro['title']}")
        pdf.add_link(f"https://youtu.be/{intro['id']}", "Watch on YouTube")
        pdf.section_body(intro['summary_and_topics'])

    # Top Videos
    pdf.section_title("Top 5 Videos")
    for i, v in enumerate(top_videos, 1):
        pdf.section_body(f"{i}. {v['title']} ({v['views']} views, 👍 {v['likes']} likes)")
        pdf.add_link(f"https://youtu.be/{v['id']}", "Watch on YouTube")
        pdf.section_body(v['summary_and_topics'])

    # Growth Graph
    if growth_img_path and os.path.exists(growth_img_path):
        pdf.section_title("Channel Growth Graph")
        pdf.add_image(growth_img_path, w=170)

    # Save PDF
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
    stats = get_all_videos_stats(channel_id)
    if not stats or len(stats) < 2:
        return None, None
    df = pd.DataFrame(stats)
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    df = df.dropna(subset=["published"])
    if filter_days:
        now_utc = pd.Timestamp.now(tz="UTC")
        df = df[df["published"] >= (now_utc - pd.Timedelta(days=filter_days))]
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
time_filter = st.selectbox("Analyze top videos from last...", ["All time", "30 days", "90 days"])
days = None if time_filter == "All time" else (30 if time_filter == "30 days" else 90)
if not url: st.info("Enter a valid YouTube link to begin."); st.stop()
vid_id, ch_id = extract_video_id(url), extract_channel_id(url)

pdf_ready = False
pdf_file_path = None
growth_img_path = None
banner_path = None
pp_path = None
growth_df = None
top_videos_full = []

if ch_id:
    details = get_channel_details(ch_id)
    if details:
        # Download images for PDF
        if details['pp']:
            pp_path = os.path.join(ASSET_PDF_PATH, "pp_temp.jpg")
            download_image(details['pp'], pp_path)
            st.image(pp_path, width=100)
        if details['banner']:
            banner_path = os.path.join(ASSET_PDF_PATH, "banner_temp.jpg")
            download_image(details['banner'], banner_path)
            st.image(banner_path, use_column_width=True)
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
        with st.expander("Show summary", expanded=True):
            st.info(best['summary_and_topics'])
        with st.expander("Show comments", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Top 3 Comments (Top comments)", key="best_top_comments_btn"):
                    if best["top_comments"]:
                        st.markdown("**Top 3 Comments (Top comments):**")
                        for c in best["top_comments"]:
                            st.write(f"👍 {c['likeCount']} - {c['text']}")
                    else:
                        st.info("No top comments found.")
            with col2:
                if st.button("Top 3 Negative Comments", key="best_neg_comments_btn"):
                    if best["neg_comments"]:
                        st.markdown("**Top 3 Negative Comments:**")
                        for c in best["neg_comments"]:
                            st.write(f"👍 {c['likeCount']} - {c['text']}")
                    else:
                        st.info("No negative comments found.")
        intro_id = details['intro_video']
        intro = get_video_summary(intro_id) if intro_id else None
        if intro:
            st.markdown("### Intro Video")
            st.video(f"https://youtu.be/{intro_id}")
            st.info(intro['summary_and_topics'])
        st.markdown("### Top 5 Videos")
        top_videos_stats = get_top_videos_stats(ch_id, 5, days=days)
        top_videos_full = []
        for i, vstat in enumerate(top_videos_stats, 1):
            v = get_video_summary(vstat['id'])
            top_videos_full.append(v)
            st.write(f"{i}. [{v['title']}](https://youtu.be/{v['id']}) - {v['views']} views, 👍 {v['likes']} likes")
            with st.expander("Show summary", expanded=False):
                st.info(v['summary_and_topics'])
            with st.expander("Show comments", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Top 3 Comments (Top comments) {i}", key=f"top_comments_btn_{i}"):
                        if v["top_comments"]:
                            st.markdown("**Top 3 Comments (Top comments):**")
                            for c in v["top_comments"]:
                                st.write(f"👍 {c['likeCount']} - {c['text']}")
                        else:
                            st.info("No top comments found.")
                with col2:
                    if st.button(f"Top 3 Negative Comments {i}", key=f"neg_comments_btn_{i}"):
                        if v["neg_comments"]:
                            st.markdown("**Top 3 Negative Comments:**")
                            for c in v["neg_comments"]:
                                st.write(f"👍 {c['likeCount']} - {c['text']}")
                        else:
                            st.info("No negative comments found.")
            if st.button("🔊 Voice Summary", key=f"voice_{v['id']}"):
                tts_playback(v['summary_and_topics'])
        st.markdown("### 📊 Channel Growth Graph")
        growth_df, growth_text = growth_estimator(ch_id, filter_days=growth_days, year_filter=growth_year)
        if growth_df is not None:
            growth_img_path = os.path.join(ASSET_PDF_PATH, "growth_temp.png")
            plot_growth_graph(growth_df, growth_img_path)
            st.line_chart(growth_df.set_index("published")["views"])
            st.write(growth_text)
            save_df_to_csv(growth_df, "growth_data.csv")
        else:
            st.info("Not enough data for growth estimation or no videos in selected period.")
        st.markdown("### SEO Analyzer")
        for vstat in top_videos_stats:
            seo = seo_analyze(vstat["title"])
            st.write(f"**{vstat['title']}** - SEO Score: {seo['score']}/{seo['max_score']}")
        st.markdown("### 🤖 Channel Q&A Chatbot")
        if "chat_history" not in st.session_state: st.session_state.chat_history = []
        user_q = st.text_input("Ask about this channel or its videos:")
        if user_q:
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

    # --- PDF GENERATION BUTTON ---
    st.markdown("---")
    if st.button("📄 Generate Final PDF Report"):
        pdf_file_path = generate_final_pdf(
            details, ch_summary_and_topics, best, intro, top_videos_full,
            growth_img_path, banner_path, pp_path,
            filename=f"{clean_text(details['title']).replace(' ','_')}_report.pdf",
            growth_df=growth_df
        )
        pdf_ready = True
    if pdf_ready and pdf_file_path and os.path.exists(pdf_file_path):
        with open(pdf_file_path, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name=os.path.basename(pdf_file_path))

elif vid_id and not ch_id:
    st.subheader("🎥 Single Video Summary")
    video = get_video_summary(vid_id)
    st.write(f"**Title:** {video['title']} ({video['views']} views, 👍 {video['likes']} likes)")
    with st.expander("Show summary", expanded=True):
        st.info(video['summary_and_topics'])
    with st.expander("Show comments", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Top 3 Comments (Top comments)", key="top_comments_btn"):
                if video["top_comments"]:
                    st.markdown("**Top 3 Comments (Top comments):**")
                    for c in video["top_comments"]:
                        st.write(f"👍 {c['likeCount']} - {c['text']}")
                else:
                    st.info("No top comments found.")
        with col2:
            if st.button("Top 3 Negative Comments", key="neg_comments_btn"):
                if video["neg_comments"]:
                    st.markdown("**Top 3 Negative Comments:**")
                    for c in video["neg_comments"]:
                        st.write(f"👍 {c['likeCount']} - {c['text']}")
                else:
                    st.info("No negative comments found.")
    if st.button("🔊 Voice Summary"):
        tts_playback(video['summary_and_topics'])
    st.markdown("---")
    if st.button("📄 Generate Final PDF Report"):
        pdf = ColorfulPDF()
        pdf.title = f"{video['title']} Video Report"
        pdf.add_page()
        pdf.section_title("Video Info")
        pdf.section_body(f"Title: {video['title']}\nViews: {video['views']}\nLikes: {video['likes']}")
        pdf.add_link(f"https://youtu.be/{video['id']}", "Watch on YouTube")
        pdf.section_title("Summary")
        pdf.section_body(video['summary_and_topics'])
        pdf.section_title("Top Comments")
        for c in video["top_comments"]:
            pdf.section_body(f"👍 {c['likeCount']} - {c['text']}")
        pdf.section_title("Top Negative Comments")
        for c in video["neg_comments"]:
            pdf.section_body(f"👍 {c['likeCount']} - {c['text']}")
        pdf_file_path = os.path.join(ASSET_PDF_PATH, f"{clean_text(video['title']).replace(' ','_')}_video_report.pdf")
        pdf.output(pdf_file_path)
        pdf_ready = True
    if pdf_ready and pdf_file_path and os.path.exists(pdf_file_path):
        with open(pdf_file_path, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name=os.path.basename(pdf_file_path))
else:
    st.warning("❌ Could not identify the channel from the URL.")