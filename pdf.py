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