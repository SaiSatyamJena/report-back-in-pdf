# D:\repos\report-back-in-pdf\mcp_reporter_app\tools_clean.py

# Standard Library Imports
import os
import shutil
import time
import json
from pathlib import Path
import asyncio

# Third-Party Imports
from dotenv import load_dotenv
import requests
from openai import OpenAI
from fpdf import FPDF
from fpdf.enums import Align
from PIL import Image as PILImage
from duckduckgo_search import DDGS

# --- Configuration and Constants ---
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
GENERATED_PDF_DIR = PROJECT_ROOT / "generated_reports"
TEMP_IMAGE_DIR = BASE_DIR / "temp_downloaded_images"
MAX_PDF_PAGES = 10
FONT_FAMILY = "Helvetica"  # Use a core font to avoid DeprecationWarnings

# --- OpenAI Client Initialization ---
openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI()
        print("[OpenAI Client] OpenAI client initialized successfully.")
    except Exception as e:
        print(
            f"[OpenAI Client] Error initializing OpenAI client: {e}. OpenAI dependent tools will fail."
        )
else:
    print(
        "WARNING: OPENAI_API_KEY not found in .env file. OpenAI dependent tools will fail."
    )


class ReportPDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.page_limit_reached = False

    def header(self):
        if self.page_limit_reached:
            return
        self.set_font(FONT_FAMILY, "B", 12)
        self.ln(5)

    def footer(self):
        if self.page_limit_reached:
            return
        self.set_y(-15)
        self.set_font(FONT_FAMILY, "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, Align.C)

    def add_content_page(self):
        if self.page_no() >= MAX_PDF_PAGES:
            if not self.page_limit_reached:
                print(
                    f"[PDF Creation] Page limit of {MAX_PDF_PAGES} reached. Cannot add new page."
                )
                self.page_limit_reached = True
            return False
        self.add_page()
        return True

    def check_page_limit_before_adding(self, estimated_height_needed_mm: float) -> bool:
        if self.page_limit_reached:
            return False
        content_height_per_page = self.h - self.t_margin - self.b_margin
        if (
            self.page_no() == MAX_PDF_PAGES
            and self.get_y() + estimated_height_needed_mm > content_height_per_page
        ):
            print(
                f"[PDF Creation] Adding {estimated_height_needed_mm}mm to page {self.page_no()} "
                f"would exceed page capacity on the last allowed page. Halting."
            )
            self.page_limit_reached = True
            return False
        if self.get_y() + estimated_height_needed_mm > content_height_per_page:
            if self.page_no() + 1 > MAX_PDF_PAGES:
                print(
                    f"[PDF Creation] Adding {estimated_height_needed_mm}mm would require a new page "
                    f"(page {self.page_no() + 1}) which exceeds limit of {MAX_PDF_PAGES}. Halting."
                )
                self.page_limit_reached = True
                return False
        return True


async def search_web_for_topic(topic: str, num_results: int = 5) -> str:
    print(
        f"[Tool: search_web_for_topic] Searching for: '{topic}' (max {num_results} results)"
    )
    results_summary = f"Web search results for '{topic}':\n"
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(keywords=topic, max_results=num_results))
            if not search_results:
                results_summary += "  No results found.\n"
                print(f"[Tool: search_web_for_topic] No results found for '{topic}'.")
                return results_summary.strip()
            for i, result in enumerate(search_results):
                title = result.get("title", "No Title")
                snippet = result.get("body", "No Snippet").replace("\n", " ")
                results_summary += f"  {i + 1}. {title}\n"
                results_summary += f"     Snippet: {snippet[:250]}...\n\n"
            print(
                f"[Tool: search_web_for_topic] Found {len(search_results)} results for '{topic}'."
            )
    except Exception as e:
        error_message = f"  Error during web search for '{topic}': {str(e)}\n"
        results_summary += error_message
        print(f"[Tool: search_web_for_topic] {error_message.strip()}")
    return results_summary.strip()


async def find_image_urls_for_topic(topic: str, num_images: int = 3) -> list[dict]:
    print(
        f"[Tool: find_image_urls_for_topic] Searching for {num_images} images for: '{topic}'"
    )
    image_info_list = []
    try:
        with DDGS() as ddgs:
            ddgs_image_gen = ddgs.images(
                keywords=topic,
                max_results=num_images + 5,
            )
            if not ddgs_image_gen:
                print(
                    f"[Tool: find_image_urls_for_topic] No image generator for '{topic}'."
                )
                return []
            count = 0
            for img_result in ddgs_image_gen:
                if count >= num_images:
                    break
                if img_result and "image" in img_result and "title" in img_result:
                    image_url = img_result["image"]
                    image_title = img_result["title"]
                    if image_url.startswith("http") and any(
                        image_url.lower().endswith(ext)
                        for ext in [".jpg", ".jpeg", ".png", ".gif"]
                    ):
                        image_info_list.append({"url": image_url, "title": image_title})
                        print(
                            f"[Tool: find_image_urls_for_topic] Found: {image_title[:50]}... URL: {image_url[:60]}..."
                        )
                        count += 1
                    else:
                        print(
                            f"[Tool: find_image_urls_for_topic] Skipping invalid/non-direct image URL: {image_url}"
                        )
            if not image_info_list:
                print(
                    f"[Tool: find_image_urls_for_topic] No suitable image URLs found for '{topic}'."
                )
    except Exception as e:
        print(
            f"[Tool: find_image_urls_for_topic] Error searching images for '{topic}': {e}"
        )
    return image_info_list[:num_images]


async def download_images(image_data_list: list[dict]) -> list[dict]:
    if not image_data_list:
        print("[Tool: download_images] No image data provided.")
        return []
    print(
        f"[Tool: download_images] Attempting to download {len(image_data_list)} image(s)."
    )
    downloaded_image_info = []
    TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for item in TEMP_IMAGE_DIR.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    for i, image_data in enumerate(image_data_list):
        image_url = image_data.get("url")
        original_title = image_data.get("title", f"image_{i}")
        if not image_url:
            print(f"[Tool: download_images] Skipping item {i} (missing URL).")
            continue
        try:
            print(f"[Tool: download_images] Downloading: {image_url}")
            response = requests.get(
                image_url, stream=True, timeout=15, allow_redirects=True
            )
            response.raise_for_status()
            file_extension = Path(image_url).suffix.split("?")[0]
            if not file_extension or len(file_extension) > 5:
                content_type = response.headers.get("content-type")
                if content_type and "image/" in content_type:
                    file_extension = (
                        "." + content_type.split("/")[-1].split(";")[0].lower()
                    )
                else:
                    file_extension = ".jpg"
            safe_title = "".join(
                c for c in original_title[:30] if c.isalnum() or c in " _-"
            ).replace(" ", "_")
            timestamp = int(time.time() * 1000)
            local_filename = f"{safe_title}_{timestamp}{file_extension}"
            local_image_path = TEMP_IMAGE_DIR / local_filename
            with open(local_image_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[Tool: download_images] Saved to: {local_image_path}")
            downloaded_image_info.append(
                {"local_path": str(local_image_path), "original_title": original_title}
            )
        except requests.exceptions.RequestException as e:
            print(f"[Tool: download_images] Failed to download {image_url}. Error: {e}")
        except IOError as e:
            print(
                f"[Tool: download_images] Failed to save image from {image_url}. IO Error: {e}"
            )
        except Exception as e:
            print(f"[Tool: download_images] Unexpected error for {image_url}: {e}")
    if not downloaded_image_info:
        print("[Tool: download_images] No images were successfully downloaded.")
    return downloaded_image_info


async def generate_report_text_content(
    topic: str,
    web_search_summary: str,
    downloaded_images_info: list[dict],
    max_pages_target: int = MAX_PDF_PAGES,
) -> dict:
    print(
        f"[Tool: generate_report_text_content] Generating report for topic: '{topic}'"
    )
    if not openai_client:
        error_msg = "OpenAI client not initialized. Cannot generate content."
        print(f"[Tool: generate_report_text_content] {error_msg}")
        return {
            "title": f"Error: {error_msg} for {topic}",
            "introduction": "",
            "sections": [],
            "conclusion": "",
        }
    available_images_text = "No specific images available to mention."
    if downloaded_images_info:
        available_images_text = "The following images have been prepared and can be referenced if relevant:\n"
        for idx, img_info in enumerate(downloaded_images_info):
            available_images_text += (
                f"- Image {idx + 1}: '{img_info.get('original_title', 'Untitled')}'\n"
            )
    prompt_messages = [
        {
            "role": "system",
            "content": f"You are an AI assistant creating a concise, informative report. "
            f"Structure: title, introduction, 2-4 main sections (heading & text), conclusion. "
            f"Target length suitable for {max_pages_target - 2}-{max_pages_target} pages including some images. "
            f"Focus on clarity and factual accuracy based on the web summary. "
            f"Do NOT include image placeholders. Images will be placed later. "
            f"Respond ONLY with a JSON object in the specified format.",
        },
        {
            "role": "user",
            "content": f"""Generate report content for: "{topic}"
Web Search Summary:
---
{web_search_summary}
---
{available_images_text}
JSON Output Format:
{{
  "title": "Report Title",
  "introduction": "Intro text...",
  "sections": [
    {{"heading": "Section 1 Heading", "text": "Section 1 content..."}},
    {{"heading": "Section 2 Heading", "text": "Section 2 content..."}}
  ],
  "conclusion": "Conclusion text..."
}}
""",
        },
    ]
    try:
        print("[Tool: generate_report_text_content] Sending request to OpenAI API...")
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=prompt_messages,
            response_format={"type": "json_object"},
            temperature=0.6,
        )
        content_json_str = response.choices[0].message.content
        print("[Tool: generate_report_text_content] Received response from OpenAI.")
        report_data = json.loads(content_json_str)
        if not all(
            k in report_data
            for k in ["title", "introduction", "sections", "conclusion"]
        ):
            raise ValueError("OpenAI response missing required keys.")
        if not isinstance(report_data["sections"], list):
            raise ValueError("'sections' must be a list.")
        print("[Tool: generate_report_text_content] Successfully parsed JSON response.")
        return report_data
    except Exception as e:
        error_message = f"Error generating report content with OpenAI: {e}"
        print(f"[Tool: generate_report_text_content] {error_message}")
        return {
            "title": f"Error Generating Report for {topic}",
            "introduction": error_message,
            "sections": [],
            "conclusion": "Content generation failed.",
        }


async def create_report_pdf(
    report_content: dict,
    downloaded_images_info: list[dict],
) -> str:
    report_title_text = report_content.get("title", "Untitled Report")
    print(f"[Tool: create_report_pdf] Creating PDF for: '{report_title_text}'")

    if not report_content or not report_title_text or "Error" in report_title_text:
        return "Error: Invalid report content or title for PDF generation."

    GENERATED_PDF_DIR.mkdir(parents=True, exist_ok=True)
    safe_report_title = "".join(
        c for c in report_title_text[:50] if c.isalnum() or c in " _-"
    ).replace(" ", "_")
    timestamp = int(time.time())
    pdf_filename = f"{safe_report_title}_{timestamp}.pdf"
    pdf_filepath = GENERATED_PDF_DIR / pdf_filename

    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    # --- Title Page ---
    if not pdf.add_content_page():
        return f"Error: PDF generation stopped at page limit (Title)."
    pdf.set_font(FONT_FAMILY, "B", 22)
    pdf.multi_cell(
        w=0, h=20, text=report_title_text, border=0, align=Align.C, ln=0
    )  # ln=0 is fine here due to explicit pdf.ln() after
    pdf.ln(20)

    page_content_width = pdf.w - pdf.l_margin - pdf.r_margin

    # --- Introduction ---
    if not pdf.page_limit_reached:
        if pdf.get_y() + 20 > (pdf.h - pdf.b_margin):
            if not pdf.add_content_page():
                return str(pdf_filepath.resolve())
        if not pdf.page_limit_reached:
            pdf.set_font(FONT_FAMILY, "B", 14)
            # ******** FIX HERE: ln=1 for heading ********
            pdf.multi_cell(
                w=0, h=10, text="Introduction", border=0, align=Align.L, ln=1
            )
            pdf.set_font(FONT_FAMILY, "", 10)
            pdf.multi_cell(
                w=0, h=6, text=report_content.get("introduction", "N/A")
            )  # Default ln=0 for content is fine
            pdf.ln(5)

    # --- Sections and Images ---
    images_iter = iter(downloaded_images_info)
    for section_idx, section in enumerate(report_content.get("sections", [])):
        if pdf.page_limit_reached:
            break

        if pdf.get_y() + 10 > (pdf.h - pdf.b_margin):
            if not pdf.add_content_page():
                break
        if pdf.page_limit_reached:
            break
        pdf.set_font(FONT_FAMILY, "B", 12)
        # ******** FIX HERE: ln=1 for heading ********
        pdf.multi_cell(
            w=0,
            h=10,
            text=section.get("heading", f"Section {section_idx + 1}"),
            border=0,
            align=Align.L,
            ln=1,
        )

        if pdf.page_limit_reached:
            break
        pdf.set_font(FONT_FAMILY, "", 10)
        pdf.multi_cell(
            w=0, h=6, text=section.get("text", "N/A")
        )  # Default ln=0 for content is fine
        pdf.ln(5)

        if not pdf.page_limit_reached:
            try:
                img_info = next(images_iter, None)
                if img_info and img_info.get("local_path"):
                    img_path_str = img_info["local_path"]
                    img_title = img_info.get("original_title", "Image")
                    if not Path(img_path_str).exists():
                        print(
                            f"[PDF Creation] Image file not found: {img_path_str}. Skipping."
                        )
                        continue
                    print(f"[PDF Creation] Processing image: {img_path_str}")
                    try:
                        with PILImage.open(img_path_str) as pil_img:
                            w_px, h_px = pil_img.size
                        img_aspect_ratio = h_px / w_px
                        render_w_mm = page_content_width * 0.7
                        render_h_mm = render_w_mm * img_aspect_ratio
                        max_img_h_mm = page_content_width * 0.6
                        if render_h_mm > max_img_h_mm:
                            render_h_mm = max_img_h_mm
                            render_w_mm = render_h_mm / img_aspect_ratio
                        if pdf.get_y() + render_h_mm + 10 > (pdf.h - pdf.b_margin):
                            if not pdf.add_content_page():
                                break
                        if pdf.page_limit_reached:
                            break
                        x_pos = (pdf.w - render_w_mm) / 2
                        pdf.image(img_path_str, x=x_pos, w=render_w_mm, h=render_h_mm)
                        pdf.set_font(FONT_FAMILY, "I", 8)
                        pdf.multi_cell(
                            w=0,
                            h=5,
                            text=f"Figure: {img_title[:80]}",
                            border=0,
                            align=Align.C,
                            ln=1,
                        )  # Caption should move to next line
                        pdf.ln(5)  # Extra space after caption + its line break
                    except FileNotFoundError:
                        print(
                            f"[PDF Creation] PIL Error: Image not found at {img_path_str}"
                        )
                    except Exception as pil_e:
                        print(
                            f"[PDF Creation] PIL Error for {img_path_str}: {pil_e}. Skipping image."
                        )
            except StopIteration:
                pass
            except Exception as e_img:
                print(f"[PDF Creation] Error handling image: {e_img}")

    # --- Conclusion ---
    if not pdf.page_limit_reached:
        if pdf.get_y() + 20 > (pdf.h - pdf.b_margin):
            if not pdf.add_content_page():
                return str(pdf_filepath.resolve())
        if not pdf.page_limit_reached:
            pdf.set_font(FONT_FAMILY, "B", 14)
            # ******** FIX HERE: ln=1 for heading ********
            pdf.multi_cell(w=0, h=10, text="Conclusion", border=0, align=Align.L, ln=1)
            pdf.set_font(FONT_FAMILY, "", 10)
            pdf.multi_cell(
                w=0, h=6, text=report_content.get("conclusion", "N/A")
            )  # Default ln=0 is fine
            pdf.ln(5)

    try:
        pdf.output(pdf_filepath, "F")
        final_msg = f"[Tool: create_report_pdf] PDF generated: {pdf_filepath.resolve()}"
        if pdf.page_limit_reached:
            final_msg += (
                f" (Note: Content may be truncated due to {MAX_PDF_PAGES} page limit)"
            )
        print(final_msg)
        return str(pdf_filepath.resolve())
    except Exception as e:
        error_message = f"Error during PDF output: {e}"
        print(f"[Tool: create_report_pdf] {error_message}")
        return f"Error: {error_message}"


async def run_full_report_generation_test(
    topic: str, num_search_results: int = 3, num_images: int = 2
):
    print(f"\n--- STARTING FULL REPORT GENERATION TEST FOR TOPIC: '{topic}' ---")
    print("\n[Step 1] Searching web for topic content...")
    web_summary = await search_web_for_topic(topic, num_results=num_search_results)
    if not web_summary or "Error" in web_summary:
        print("  Failed to get web summary. Aborting test.")
        return
    print(f"  Web Summary (first 100 chars): {web_summary[:100]}...")
    print("\n[Step 2] Finding image URLs...")
    image_urls_data = await find_image_urls_for_topic(topic, num_images=num_images)
    if not image_urls_data:
        print("  No image URLs found. Proceeding without images for PDF.")
        downloaded_images_info = []
    else:
        print(f"  Found {len(image_urls_data)} image URLs.")
        print("\n[Step 3] Downloading images...")
        downloaded_images_info = await download_images(image_urls_data)
        if not downloaded_images_info:
            print("  Failed to download any images. Proceeding without images for PDF.")
        else:
            print(f"  Successfully downloaded {len(downloaded_images_info)} images.")
            for img_info in downloaded_images_info:
                print(
                    f"    - Downloaded: {img_info['original_title']} to {img_info['local_path']}"
                )
    if not openai_client:
        print(
            "\n[Step 4] Skipping report text generation (OpenAI client not available)."
        )
        report_content = {
            "title": f"Mock Report: {topic} (OpenAI Not Used)",
            "introduction": "This is a mock introduction as OpenAI client was not available. "
            "The report demonstrates PDF creation with available data.\n"
            + (web_summary.splitlines()[0] if web_summary else ""),
            "sections": [
                {
                    "heading": "Mock Section 1",
                    "text": "Content for mock section 1 based on web summary:\n"
                    + (
                        web_summary.splitlines()[1][:200]
                        if web_summary and len(web_summary.splitlines()) > 2
                        else "Mock text for section 1."
                    ),
                },
                {
                    "heading": "Mock Section 2",
                    "text": "This section would normally contain AI-generated content. Here's some more mock text to fill space and test layout with multiple paragraphs. This ensures that wrapping and line spacing are handled correctly even for longer text blocks.",
                },
            ],
            "conclusion": "This mock conclusion ends the report. It summarizes the mock findings and aims to test the final parts of the PDF layout. Hopefully, everything looks good!",
        }
        print(f"  Using mock content for PDF. Title: {report_content['title']}")
    else:
        print("\n[Step 4] Generating report text content with OpenAI...")
        report_content = await generate_report_text_content(
            topic=topic,
            web_search_summary=web_summary,
            downloaded_images_info=downloaded_images_info,
            max_pages_target=MAX_PDF_PAGES,
        )
        if "Error" in report_content.get("title", "") or not report_content.get(
            "introduction"
        ):  # Check if content is valid
            print(
                f"  Failed to generate report content or content is invalid: {report_content.get('introduction', 'Unknown OpenAI error or empty content')}"
            )
            print("  Aborting PDF creation.")
            return
        print(f"  Report content generated. Title: {report_content['title']}")
    print("\n[Step 5] Creating PDF report...")
    pdf_path_or_error = await create_report_pdf(
        report_content=report_content, downloaded_images_info=downloaded_images_info
    )
    print("\n--- PDF CREATION RESULT ---")
    if isinstance(pdf_path_or_error, str) and "Error" in pdf_path_or_error:
        print(f"  PDF Creation Failed: {pdf_path_or_error}")
    elif isinstance(pdf_path_or_error, str) and Path(pdf_path_or_error).exists():
        print(f"  SUCCESS: PDF report generated at: {pdf_path_or_error}")
        print(f"  Please check the folder: {GENERATED_PDF_DIR.resolve()}")
    else:
        print(f"  PDF creation returned an unexpected result: {pdf_path_or_error}")
    print("\n--- FULL REPORT GENERATION TEST COMPLETED ---")


if __name__ == "__main__":
    search_topic = "advancements in quantum mechanics"
    num_web_results_for_test = 3
    num_images_for_test = 2
    asyncio.run(
        run_full_report_generation_test(
            topic=search_topic,
            num_search_results=num_web_results_for_test,
            num_images=num_images_for_test,
        )
    )
