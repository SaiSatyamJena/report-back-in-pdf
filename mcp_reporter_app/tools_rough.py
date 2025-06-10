# D:\repos\report-back-in-pdf\mcp_reporter_app\tools.py

import os
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import shutil
from pathlib import Path
import time
from openai import OpenAI
from fpdf import FPDF  # Ensure this is imported at the top
from fpdf.enums import Align  # For aligning text and images (newer FPDF2 feature)
from PIL import Image as PILImage  # For getting image dimensions to help with placement

# Load environment variables from .env file located in the parent directory
# This needs to be done early, especially if other imports rely on these vars
# Construct the path to the .env file relative to this tools.py file
# This should also be in .gitignore if we don't want to commit generated PDFs.
GENERATED_PDF_DIR = (
    Path(__file__).parent.parent / "generated_reports"
)  # Project_root / generated_reports
MAX_PDF_PAGES = 10  # Our defined constraint

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)
# --- Environment Variable Check (Optional but good for early feedback) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print(
        "WARNING: OPENAI_API_KEY not found in .env file. OpenAI dependent tools will fail."
    )
else:
    print("OpenAI API Key successfully loaded from .env (from tools.py).")

# This directory should ideally be in .gitignore if it's purely temporary for a session
# For now, let's create it inside mcp_reporter_app, but we'll ensure it's cleaned up or ignored.
# A better approach for production might be a system temp directory,
# but for this MVP, a local sub-folder is simpler to manage.
TEMP_IMAGE_DIR = Path(__file__).parent / "temp_downloaded_images"


class ReportPDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.page_limit_reached = False  # Custom flag

    def header(self):
        if self.page_limit_reached:
            return
        # Simple header - you can customize this
        self.set_font("Arial", "B", 12)
        # self.cell(0, 10, 'Your Report Title - Can be dynamic', 0, 1, Align.C) # Centered
        self.ln(5)  # Line break

    def footer(self):
        if self.page_limit_reached:
            return
        # Simple footer - page number
        self.set_y(-15)  # Position 1.5 cm from bottom
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, Align.C)

    def add_content_page(self):
        if self.page_no() >= MAX_PDF_PAGES:
            if not self.page_limit_reached:
                print(
                    f"[PDF Creation] Page limit of {MAX_PDF_PAGES} reached or exceeded. Halting content addition."
                )
                self.page_limit_reached = True
            return False  # Indicate page limit reached
        self.add_page()
        return True

    def check_page_limit_before_adding(self, estimated_height_needed_mm):
        """Checks if adding content of certain height would exceed page limit"""
        if self.page_limit_reached:
            return False  # Already over limit

        # Effective page height for content (A4 height 297mm - top/bottom margins)
        # Assuming default margins of 10mm top/bottom for simplicity (fpdf default can vary)
        # Or get from self.t_margin, self.b_margin
        content_height_per_page = self.h - self.t_margin - self.b_margin

        # If current y + needed height > page content height, AND we are on the last allowed page
        if (self.get_y() + estimated_height_needed_mm > content_height_per_page) and (
            self.page_no() >= MAX_PDF_PAGES - 1
        ):  # -1 because we might be about to add a page that IS the limit
            print(
                f"[PDF Creation] Adding {estimated_height_needed_mm}mm would likely exceed page limit. Halting."
            )
            self.page_limit_reached = True
            return False
        return True


async def search_web_for_topic(topic: str, num_results: int = 5) -> str:
    """
    Searches the web for a given topic using DuckDuckGo and returns a formatted string
    containing titles and snippets of the search results.

    Args:
        topic (str): The topic to search for.
        num_results (int): The maximum number of search results to retrieve.

    Returns:
        str: A string summarizing the search results, or an error message.
    """
    print(
        f"[Tool: search_web_for_topic] Searching for: '{topic}' (max {num_results} results)"
    )
    results_summary = f"Web search results for '{topic}':\n"

    try:
        with DDGS() as ddgs:
            # .text() method for general web search results
            search_results = list(ddgs.text(keywords=topic, max_results=num_results))

            if not search_results:
                results_summary += "  No results found.\n"
                print(f"[Tool: search_web_for_topic] No results found for '{topic}'.")
                return results_summary.strip()

            for i, result in enumerate(search_results):
                title = result.get("title", "No Title")
                snippet = result.get("body", "No Snippet").replace(
                    "\n", " "
                )  # Ensure snippet is single line
                # href = result.get('href', '#') # URL of the result
                results_summary += f"  {i + 1}. {title}\n"
                results_summary += f"     Snippet: {snippet[:250]}...\n\n"  # Limit snippet length for summary

            print(
                f"[Tool: search_web_for_topic] Found {len(search_results)} results for '{topic}'."
            )

    except Exception as e:
        error_message = f"  Error during web search for '{topic}': {str(e)}\n"
        results_summary += error_message
        print(f"[Tool: search_web_for_topic] {error_message.strip()}")
        # Depending on desired behavior, you might want to raise the exception
        # or just return the error message within the summary.
        # For this tool, returning it in the summary is fine for now.

    return results_summary.strip()


async def find_image_urls_for_topic(topic: str, num_images: int = 3) -> list[dict]:
    """
    Searches for image URLs related to a given topic using DuckDuckGo.
    Returns a list of dictionaries, each containing 'url' and 'title' for an image.

    Args:
        topic (str): The topic to search images for.
        num_images (int): The desired number of image URLs to find.

    Returns:
        list[dict]: A list of dictionaries, e.g., [{'url': '...', 'title': '...'}, ...],
                    or an empty list if no images are found or an error occurs.
    """
    print(
        f"[Tool: find_image_urls_for_topic] Searching for {num_images} images related to: '{topic}'"
    )
    image_info_list = []

    try:
        with DDGS() as ddgs:
            # ddgs.images returns a generator of image results
            # We'll fetch a bit more than num_images because some might be unusable later
            ddgs_image_gen = ddgs.images(
                keywords=topic,
                region="wt-wt",  # Worldwide
                safesearch="moderate",  # Or 'off' or 'strict'
                size=None,  # Any size
                # type_image=None,   # Any type (photo, clipart, gif, etc.)
                # layout=None,       # Any layout (square, tall, wide)
                # license_image=None, # Any license (we are not filtering by license for this MVP)
                max_results=num_images + 5,  # Fetch a few extra
            )

            if not ddgs_image_gen:
                print(
                    f"[Tool: find_image_urls_for_topic] DDGS images generator is empty for '{topic}'."
                )
                return image_info_list

            count = 0
            for img_result in ddgs_image_gen:
                if count >= num_images:
                    break
                if img_result and "image" in img_result and "title" in img_result:
                    image_url = img_result["image"]
                    image_title = img_result["title"]

                    # Basic validation: ensure it looks like an image URL
                    # More robust validation would involve trying to fetch headers or content type
                    if image_url.startswith("http") and any(
                        image_url.lower().endswith(ext)
                        for ext in [".jpg", ".jpeg", ".png", ".gif"]
                    ):
                        image_info_list.append({"url": image_url, "title": image_title})
                        print(
                            f"[Tool: find_image_urls_for_topic] Found image: {image_title} - {image_url}"
                        )
                        count += 1
                    else:
                        print(
                            f"[Tool: find_image_urls_for_topic] Skipping invalid or non-direct image URL: {image_url}"
                        )
                else:
                    # This case should be rare if ddgs.images behaves as expected
                    print(
                        f"[Tool: find_image_urls_for_topic] Received an unexpected image result structure: {img_result}"
                    )

            if not image_info_list:
                print(
                    f"[Tool: find_image_urls_for_topic] No suitable image URLs found for '{topic}'."
                )

    except Exception as e:
        print(
            f"[Tool: find_image_urls_for_topic] Error during image search for '{topic}': {str(e)}"
        )
        # For this tool, we'll return an empty list on error.
        # In a more robust system, you might raise the error or handle it differently.

    return image_info_list[:num_images]  # Return only the number of images requested


# --- Simple Test (optional, for direct execution of this file) ---
# if __name__ == "__main__":
#     import asyncio

#     async def main_test():
#         # ... (existing test for search_web_for_topic can remain) ...

#         # Test the find_image_urls_for_topic function
#         image_test_topic = "beautiful landscapes"
#         print(
#             f"\n--- Testing find_image_urls_for_topic with topic: '{image_test_topic}' ---"
#         )
#         image_urls_data = await find_image_urls_for_topic(
#             image_test_topic, num_images=2
#         )

#         if image_urls_data:
#             print("\n--- Found Image URLs & Titles ---")
#             for item in image_urls_data:
#                 print(f"  Title: {item['title']}")
#                 print(f"  URL: {item['url']}")
#         else:
#             print("  No image URLs found.")
#         print("--- End of Image Test ---")

#     # Make sure load_dotenv is called if you run this standalone and it's not at the top level
#     # dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # Recalculate if needed
#     # load_dotenv(dotenv_path=dotenv_path)
#     asyncio.run(main_test())

# --- Simple Test (optional, for direct execution of this file) ---
# if __name__ == "__main__":
#     import asyncio

#     async def main_test():
#         # Test the search_web_for_topic function
#         test_topic = "latest advancements in quantum computing"
#         print(f"--- Testing search_web_for_topic with topic: '{test_topic}' ---")
#         summary = await search_web_for_topic(test_topic, num_results=3)
#         print("\n--- Search Summary ---")
#         print(summary)
#         print("--- End of Test ---")


#     asyncio.run(main_test())
async def download_images(image_data_list: list[dict]) -> list[dict]:
    """
    Downloads images from a list of URLs and saves them to a local temporary directory.

    Args:
        image_data_list (list[dict]): A list of dictionaries, where each dict
                                      is expected to have a 'url' key with the image URL
                                      and a 'title' key.
                                      Example: [{'url': '...', 'title': '...'}, ...]

    Returns:
        list[dict]: A list of dictionaries for successfully downloaded images,
                    each containing 'local_path' (str to the saved image)
                    and 'original_title' (str).
                    Returns an empty list if no images are successfully downloaded.
    """
    if not image_data_list:
        print("[Tool: download_images] No image data provided to download.")
        return []

    print(
        f"[Tool: download_images] Attempting to download {len(image_data_list)} image(s)."
    )
    downloaded_image_info = []

    # Ensure the temporary directory exists
    TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Optional: Clean up old images from the directory if it's meant to be fresh each run
    for item in TEMP_IMAGE_DIR.iterdir():
        if item.is_file():
            item.unlink()  # Delete file
        elif item.is_dir():
            shutil.rmtree(item)  # Delete directory recursively

    for i, image_data in enumerate(image_data_list):
        image_url = image_data.get("url")
        original_title = image_data.get("title", f"image_{i}")

        if not image_url:
            print(f"[Tool: download_images] Skipping item {i} due to missing URL.")
            continue

        try:
            print(f"[Tool: download_images] Downloading: {image_url}")
            # Make the HTTP GET request to fetch the image
            # Added a timeout to prevent hanging indefinitely
            # Allow redirects as some image URLs might be behind shorteners or CDNs
            response = requests.get(
                image_url, stream=True, timeout=10, allow_redirects=True
            )
            response.raise_for_status()  # Raise an HTTPError for bad responses (4XX or 5XX)

            # Create a unique-ish filename to avoid collisions
            # Get the file extension from the URL (simple approach)
            file_extension = Path(image_url).suffix.split("?")[
                0
            ]  # Handle URLs with query params
            if (
                not file_extension or len(file_extension) > 5
            ):  # Basic check for valid extension
                # Try to get extension from content-type if header is present
                content_type = response.headers.get("content-type")
                if content_type and "image/" in content_type:
                    file_extension = (
                        "." + content_type.split("/")[-1].split(";")[0]
                    )  # e.g. .jpeg, .png
                else:
                    file_extension = ".jpg"  # Default if no clear extension

            # Sanitize original_title for use in filename (basic)
            safe_title = "".join(
                c if c.isalnum() or c in (" ", "_", "-") else "_"
                for c in original_title[:30]
            )
            safe_title = safe_title.replace(" ", "_")

            # Use timestamp to help ensure uniqueness if titles are very similar
            timestamp = int(time.time() * 1000)
            local_filename = f"{safe_title}_{timestamp}{file_extension}"
            local_image_path = TEMP_IMAGE_DIR / local_filename

            # Write the content to the local file
            with open(local_image_path, "wb") as f:
                # shutil.copyfileobj(response.raw, f) # Good for large files
                for chunk in response.iter_content(
                    chunk_size=8192
                ):  # Iterate over chunks
                    f.write(chunk)

            print(
                f"[Tool: download_images] Successfully downloaded and saved to: {local_image_path}"
            )
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
            print(
                f"[Tool: download_images] An unexpected error occurred for {image_url}: {e}"
            )

    if not downloaded_image_info:
        print("[Tool: download_images] No images were successfully downloaded.")

    return downloaded_image_info


# --- Simple Test (optional, for direct execution of this file) ---
# if __name__ == "__main__":
#     import asyncio

#     async def main_test():
#         # ... (existing tests for search_web_for_topic and find_image_urls_for_topic) ...

#         # First, get some image URLs to test downloading
#         image_test_topic_dl = "cute kittens"
#         print(
#             f"\n--- Testing find_image_urls_for_topic for download test: '{image_test_topic_dl}' ---"
#         )
#         urls_to_download_data = await find_image_urls_for_topic(
#             image_test_topic_dl, num_images=2
#         )

#         if urls_to_download_data:
#             print("\n--- Image URLs found for download test ---")
#             for item in urls_to_download_data:
#                 print(f"  Title: {item['title']}, URL: {item['url']}")

#             print("\n--- Testing download_images ---")
#             # The download_images function itself is async, so we await it
#             downloaded_files_info = await download_images(urls_to_download_data)

#             if downloaded_files_info:
#                 print("\n--- Successfully Downloaded Images ---")
#                 for file_info in downloaded_files_info:
#                     print(f"  Original Title: {file_info['original_title']}")
#                     print(f"  Saved to: {file_info['local_path']}")
#                 print(f"  (Check the folder: {TEMP_IMAGE_DIR.resolve()})")
#             else:
#                 print("  No images were downloaded in the test.")
#         else:
#             print(
#                 f"  No image URLs found for '{image_test_topic_dl}' to test downloading."
#             )
#         print("--- End of Download Test ---")

#     # Make sure load_dotenv is called if you run this standalone
#     # dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
#     # load_dotenv(dotenv_path=dotenv_path)
#     asyncio.run(main_test())
# Initialize OpenAI client
# It's good practice to initialize it once if it's used by multiple functions in this module.
# The API key is automatically picked up by the OpenAI library from the environment variable if set.
try:
    openai_client = OpenAI()  # API key is read from OPENAI_API_KEY environment variable
    # Test with a simple, cheap call if you want to confirm client setup here, e.g., list models
    # models = openai_client.models.list()
    # print("[OpenAI Client] Successfully initialized and connected.")
except Exception as e:
    print(
        f"[OpenAI Client] Error initializing OpenAI client: {e}. Check API key and network."
    )
    openai_client = None  # Set to None so attempts to use it will clearly fail


async def generate_report_text_content(
    topic: str,
    web_search_summary: str,
    downloaded_images_info: list[
        dict
    ],  # List of {'local_path': '...', 'original_title': '...'}
    max_pages_target: int = 10,  # For informing the LLM about desired conciseness
) -> dict:
    """
    Uses OpenAI API to generate structured text content for a report.

    Args:
        topic (str): The main topic of the report.
        web_search_summary (str): A summary of information found from web searches.
        downloaded_images_info (list[dict]): Info about downloaded images, including 'original_title'.
                                             This helps the LLM know what images are available.
        max_pages_target (int): An indicative target for report length to guide LLM conciseness.

    Returns:
        dict: A dictionary containing 'title', 'introduction', 'sections' (a list of dicts
              with 'heading' and 'text'), and 'conclusion'.
              Returns a dict with error info if generation fails.
    """
    print(
        f"[Tool: generate_report_text_content] Generating report content for topic: '{topic}'"
    )

    if not openai_client:
        print(
            "[Tool: generate_report_text_content] OpenAI client not initialized. Cannot generate content."
        )
        return {
            "title": f"Error: OpenAI Client Not Initialized for {topic}",
            "introduction": "The OpenAI client could not be set up. Please check API key and network.",
            "sections": [],
            "conclusion": "Content generation failed due to client initialization error.",
        }

    # Prepare a textual representation of available images for the prompt
    available_images_text = "No specific images available to mention."
    if downloaded_images_info:
        available_images_text = "The following images have been prepared and can be referenced if relevant:\n"
        for idx, img_info in enumerate(downloaded_images_info):
            available_images_text += f"- Image {idx + 1}: '{img_info.get('original_title', 'Untitled Image')}'\n"

    # Construct the prompt for OpenAI
    # This prompt needs to be carefully engineered for good results.
    prompt_messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant tasked with writing a concise and informative report on a given topic. "
            f"The report should be structured with a title, an introduction, 2-4 main sections with clear headings, "
            f"and a conclusion. The total content should be suitable for a report of approximately {max_pages_target - 2}-{max_pages_target} "  # Deduct some for images, title page
            f"pages, considering some images will be included. Aim for clarity and factual accuracy based on the provided summary. "
            f"You do NOT need to include image placeholders like [IMAGE HERE]. Simply write the text content. "
            f"The available images will be placed systematically later.",
        },
        {
            "role": "user",
            "content": f"""Please generate the text content for a report on the topic: "{topic}"

Here is a summary from a web search on this topic:
--- WEB SEARCH SUMMARY ---
{web_search_summary}
--- END WEB SEARCH SUMMARY ---

{available_images_text}

Based on this information, please generate the report content with the following structure:
1.  **Title:** A compelling title for the report.
2.  **Introduction:** A brief introduction to the topic (1-2 paragraphs).
3.  **Main Sections:** 2 to 4 sections, each with a descriptive heading and informative content (a few paragraphs per section).
4.  **Conclusion:** A concise summary or concluding thoughts (1-2 paragraphs).

Respond ONLY with a JSON object in the following format:
{{
  "title": "Your Generated Report Title",
  "introduction": "Text for the introduction...",
  "sections": [
    {{ "heading": "Heading for Section 1", "text": "Content for section 1..." }},
    {{ "heading": "Heading for Section 2", "text": "Content for section 2..." }}
  ],
  "conclusion": "Text for the conclusion..."
}}
""",
        },
    ]

    try:
        print("[Tool: generate_report_text_content] Sending request to OpenAI API...")
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-4-turbo-preview" if you have access and prefer for quality
            messages=prompt_messages,
            response_format={"type": "json_object"},  # Request JSON output
            temperature=0.7,  # A balance between creativity and determinism
        )

        # The response.choices[0].message.content should be a JSON string
        content_json_str = response.choices[0].message.content
        print("[Tool: generate_report_text_content] Received response from OpenAI.")
        # print(f"Raw OpenAI response string: {content_json_str[:500]}...") # For debugging

        import json  # Import locally for this use

        report_data = json.loads(content_json_str)

        # Basic validation of the returned structure (can be more thorough)
        if not all(
            k in report_data
            for k in ["title", "introduction", "sections", "conclusion"]
        ):
            raise ValueError(
                "OpenAI response did not contain all required keys in the JSON object."
            )
        if not isinstance(report_data["sections"], list):
            raise ValueError("OpenAI response 'sections' should be a list.")

        print(
            "[Tool: generate_report_text_content] Successfully parsed JSON response from OpenAI."
        )
        return report_data

    except Exception as e:
        error_message = f"Error generating report content with OpenAI: {str(e)}"
        print(f"[Tool: generate_report_text_content] {error_message}")
        return {
            "title": f"Error Generating Report Content for {topic}",
            "introduction": error_message,
            "sections": [],
            "conclusion": "Could not generate full content due to an API error or parsing issue.",
        }


# --- Simple Test (optional, for direct execution of this file) ---
# if __name__ == "__main__":
#     import asyncio

#     async def main_test():
#         # ... (existing tests can be here) ...

#         # Test generate_report_text_content
#         # Prerequisite: Get web search summary and image info
#         test_content_topic = "the impact of AI on journalism"
#         print(
#             f"\n--- Getting prerequisites for content generation test: '{test_content_topic}' ---"
#         )

#         # 1. Get web summary
#         web_summary = await search_web_for_topic(test_content_topic, num_results=3)
#         print("\n--- Web Summary for Content Gen Test ---")
#         print(web_summary)

#         # 2. Get image info (pretend we downloaded them for this test)
#         #    In a real flow, this would come from download_images
#         mock_downloaded_images = [
#             {"local_path": "path/to/image1.jpg", "original_title": "AI and newsroom"},
#             {
#                 "local_path": "path/to/image2.png",
#                 "original_title": "Robot journalist concept",
#             },
#         ]
#         print("\n--- Mock Downloaded Image Info ---")
#         for img in mock_downloaded_images:
#             print(f"  Title: {img['original_title']}")

#         print(
#             f"\n--- Testing generate_report_text_content for topic: '{test_content_topic}' ---"
#         )
#         generated_content = await generate_report_text_content(
#             topic=test_content_topic,
#             web_search_summary=web_summary,
#             downloaded_images_info=mock_downloaded_images,  # Use mock data
#             max_pages_target=5,  # Test with a smaller target
#         )

#         print("\n--- Generated Report Content (JSON Structure) ---")
#         import json  # For pretty printing

#         print(json.dumps(generated_content, indent=2))
#         print("--- End of Content Generation Test ---")


#     # Ensure .env is loaded (should be by the top-level load_dotenv)
#     asyncio.run(main_test())
async def create_report_pdf(
    report_content: dict,  # {'title': ..., 'introduction': ..., 'sections': [...], 'conclusion': ...}
    downloaded_images_info: list[
        dict
    ],  # List of {'local_path': '...', 'original_title': '...'}
) -> str:
    """
    Creates a PDF document from the report content and downloaded images.

    Args:
        report_content (dict): The structured text content of the report.
        downloaded_images_info (list[dict]): Information about downloaded local images.

    Returns:
        str: The absolute path to the generated PDF file, or an error message string.
    """
    report_title_text = report_content.get("title", "Untitled Report")
    print(f"[Tool: create_report_pdf] Creating PDF for: '{report_title_text}'")

    if not report_content or not report_title_text:
        return "Error: Report content or title is missing for PDF generation."

    # Ensure the output directory for PDFs exists
    GENERATED_PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitize title for filename
    safe_report_title = "".join(
        c if c.isalnum() or c in (" ", "_", "-") else "_"
        for c in report_title_text[:50]
    )
    safe_report_title = safe_report_title.replace(" ", "_")
    timestamp = int(time.time())
    pdf_filename = f"{safe_report_title}_{timestamp}.pdf"
    pdf_filepath = GENERATED_PDF_DIR / pdf_filename

    pdf = ReportPDF(orientation="P", unit="mm", format="A4")  # Portrait, mm, A4
    pdf.set_auto_page_break(
        auto=True, margin=15
    )  # Auto page break with 15mm bottom margin
    pdf.set_fill_color(200, 220, 255)  # Example fill color for some elements

    # --- Title Page ---
    if not pdf.add_content_page():
        return f"Error: PDF generation stopped at page limit (Title)."
    pdf.set_font("Arial", "B", 24)
    pdf.multi_cell(0, 20, report_title_text, 0, Align.C)  # Centered title
    pdf.ln(20)  # Extra space

    # --- Introduction ---
    if not pdf.check_page_limit_before_adding(20):  # Estimate height for intro
        return str(pdf_filepath)  # Return path even if truncated
    if not pdf.page_limit_reached:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Introduction", 0, 1, Align.L)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(
            0, 6, report_content.get("introduction", "No introduction provided.")
        )
        pdf.ln(5)

    # --- Sections and Images ---
    images_iterator = iter(downloaded_images_info)  # To get next image

    for section_idx, section in enumerate(report_content.get("sections", [])):
        if pdf.page_limit_reached:
            break

        # Section Heading
        if not pdf.check_page_limit_before_adding(10):
            break  # Estimate for heading
        if not pdf.page_limit_reached:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(
                0,
                10,
                section.get("heading", f"Section {section_idx + 1}"),
                0,
                1,
                Align.L,
            )
            pdf.set_font("Arial", "", 10)

        # Section Text
        if not pdf.check_page_limit_before_adding(20):
            break  # Estimate for text
        if not pdf.page_limit_reached:
            pdf.multi_cell(0, 6, section.get("text", "No content for this section."))
            pdf.ln(5)

        # Try to add an image after this section (if available and space permits)
        if not pdf.page_limit_reached:
            try:
                image_info = next(images_iterator, None)
                if image_info and image_info.get("local_path"):
                    local_image_path_str = image_info["local_path"]
                    original_image_title = image_info.get("original_title", "Image")

                    if not Path(local_image_path_str).exists():
                        print(
                            f"[PDF Creation] Image file not found: {local_image_path_str}. Skipping."
                        )
                        continue

                    # Estimate image height (complex, this is a rough simplification)
                    # For better results, one would open image, get dimensions, scale, then estimate PDF height
                    # Assuming an image might take up 50-80mm of height
                    if not pdf.check_page_limit_before_adding(60):
                        break
                    if pdf.page_limit_reached:
                        break

                    print(
                        f"[PDF Creation] Attempting to add image: {local_image_path_str}"
                    )

                    # Get image dimensions to scale it appropriately
                    try:
                        with PILImage.open(local_image_path_str) as img:
                            img_width_px, img_height_px = img.size

                        # Convert pixels to mm (approx @ 96 DPI for web images, FPDF default 72 DPI for points)
                        # FPDF uses points by default if unit isn't mm. 1 inch = 25.4 mm = 72 points.
                        # So 1 pixel @ 96 DPI = (25.4/96) mm.
                        # For simplicity with FPDF's mm unit:
                        # Max width for image in PDF (A4 width 210mm - 2*margins)
                        pdf_content_width_mm = (
                            pdf.w - pdf.l_margin - pdf.r_margin
                        )  # pdf.w is page width in current units

                        # Scale image to fit width, maintaining aspect ratio
                        img_render_width_mm = min(
                            img_width_px * 0.2, pdf_content_width_mm * 0.8
                        )  # Heuristic: 0.2 factor, max 80% of content width
                        img_aspect_ratio = img_height_px / img_width_px
                        img_render_height_mm = img_render_width_mm * img_aspect_ratio

                        # Check if this scaled image fits the remaining page height
                        if pdf.get_y() + img_render_height_mm + 10 > (
                            pdf.h - pdf.b_margin
                        ):  # +10 for caption/spacing
                            if not pdf.add_content_page():
                                break  # Add new page if it won't exceed limit

                        if not pdf.page_limit_reached:
                            # Add image, centered using x coordinate calculation
                            x_pos = (pdf.w - img_render_width_mm) / 2
                            pdf.image(
                                local_image_path_str,
                                x=x_pos,
                                w=img_render_width_mm,
                                h=img_render_height_mm,
                            )
                            pdf.set_font("Arial", "I", 8)
                            pdf.multi_cell(
                                0, 5, f"Figure: {original_image_title[:80]}", 0, Align.C
                            )  # Centered caption
                            pdf.ln(5)

                    except FileNotFoundError:
                        print(
                            f"[PDF Creation] PIL Error: Image file not found during dimension check: {local_image_path_str}"
                        )
                    except Exception as pil_e:  # Catch other PIL errors
                        print(
                            f"[PDF Creation] PIL Error processing image {local_image_path_str}: {pil_e}. Skipping image."
                        )

            except StopIteration:
                # No more images to add
                pass
            except Exception as e_img:
                print(f"[PDF Creation] Error during image handling: {e_img}")

    # --- Conclusion ---
    if not pdf.page_limit_reached:
        if not pdf.check_page_limit_before_adding(20):  # Estimate for conclusion
            pass  # Allow saving if conclusion is cut, but don't add if it would push over
        if not pdf.page_limit_reached:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Conclusion", 0, 1, Align.L)
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(
                0, 6, report_content.get("conclusion", "No conclusion provided.")
            )
            pdf.ln(5)

    # --- Output the PDF ---
    try:
        pdf.output(pdf_filepath, "F")  # 'F' to save to local file
        print(
            f"[Tool: create_report_pdf] PDF generated successfully: {pdf_filepath.resolve()}"
        )
        return str(pdf_filepath.resolve())
    except Exception as e:
        error_message = f"Error during PDF output: {str(e)}"
        print(f"[Tool: create_report_pdf] {error_message}")
        return f"Error: {error_message}"


# --- Simple Test (optional, for direct execution of this file) ---
if __name__ == "__main__":
    import asyncio
    import json  # For loading mock data or pretty printing

    async def main_test_pdf_creation():
        print("\n--- Setting up prerequisites for PDF creation test ---")

        # 1. Mock report content (as if from generate_report_text_content)
        mock_report_content = {
            "title": "The Dynamic World of Test PDFs",
            "introduction": "This introduction serves to test the PDF generation capabilities. It will explore how text flows and images are embedded within the document, aiming for a professional and readable output that respects the defined page limits. The successful creation of this document will validate the core PDF generation logic.",
            "sections": [
                {
                    "heading": "Section One: Text Formatting",
                    "text": "Proper text formatting is essential for readability. This section demonstrates the use of different font styles, sizes, and alignments. Multi-line text will be handled using multi_cell, ensuring that paragraphs wrap correctly within the defined page margins. We will also observe how section headings stand out from the body text, contributing to the overall structure and navigability of the report.",
                },
                {
                    "heading": "Section Two: Image Handling and Placement",
                    "text": "Images can significantly enhance a report. This part focuses on embedding images, scaling them appropriately to fit the page while maintaining aspect ratio, and adding captions. The goal is to integrate images in a way that complements the text rather than disrupts it. We will use a couple of sample images downloaded specifically for this test to ensure the process is robust.",
                },
                {
                    "heading": "Section Three: Adherence to Page Limits",
                    "text": "The report is constrained to a maximum of 10 pages. This section is designed to test the page limit enforcement. If the content, including text and images, threatens to exceed this limit, the PDF generation process should gracefully stop adding new content, ensuring the final document remains within the specified bounds. This is crucial for automated systems generating numerous reports.",
                },
            ],
            "conclusion": "To conclude, this test aims to verify the complete PDF generation pipeline. From text formatting and image embedding to page limit management, all components must work harmoniously. The final PDF should be a well-structured, visually acceptable document that serves as a good proof-of-concept for the report generation tool.",
        }
        print("  Mock report content prepared.")

        # 2. Download a couple of NEW images specifically for this PDF test
        print("\n  Attempting to download test images for PDF creation...")
        test_image_topic = "abstract art"  # A topic likely to yield diverse images
        # We use find_image_urls_for_topic and then download_images
        # This makes the test more self-contained.
        image_urls_data = await find_image_urls_for_topic(
            test_image_topic, num_images=2
        )

        downloaded_test_images_info = []
        if image_urls_data:
            print(
                f"  Found {len(image_urls_data)} image URLs for '{test_image_topic}'. Downloading them now..."
            )
            downloaded_test_images_info = await download_images(image_urls_data)
            if downloaded_test_images_info:
                print(
                    f"  Successfully downloaded {len(downloaded_test_images_info)} images for the PDF test."
                )
                for img_info in downloaded_test_images_info:
                    print(
                        f"    - Title: {img_info['original_title']}, Path: {img_info['local_path']}"
                    )
            else:
                print(
                    f"  Failed to download any images for topic '{test_image_topic}' for the PDF test."
                )
        else:
            print(
                f"  Could not find any image URLs for topic '{test_image_topic}' for the PDF test."
            )

        # If no images were downloaded, the PDF will be text-only, which is also a valid test case.
        if not downloaded_test_images_info:
            print(
                "  Proceeding with PDF generation (will be text-only or have fewer images)."
            )

        print("\n--- Testing create_report_pdf ---")
        pdf_path_or_error = await create_report_pdf(
            report_content=mock_report_content,
            downloaded_images_info=downloaded_test_images_info,  # Use the newly downloaded images
        )

        print("\n--- PDF Creation Result ---")
        if (
            isinstance(pdf_path_or_error, str) and "Error" in pdf_path_or_error
        ):  # Check if it's an error string
            print(f"  Error: {pdf_path_or_error}")
        elif (
            isinstance(pdf_path_or_error, str) and Path(pdf_path_or_error).exists()
        ):  # Check if it's a valid path
            print(f"  PDF successfully generated at: {pdf_path_or_error}")
            print(f"  Please check the folder: {GENERATED_PDF_DIR.resolve()}")
        else:
            print(f"  An unexpected result was returned: {pdf_path_or_error}")

        print("--- End of PDF Creation Test ---")

        # Optional: Cleanup the specifically downloaded test images if needed.
        # However, if your download_images uses a general temp folder that's auto-cleaned,
        # this might be redundant or could interfere if the temp dir is shared.
        # For now, we rely on the TEMP_IMAGE_DIR possibly being cleaned elsewhere or manually.
        # If TEMP_IMAGE_DIR is specific to a single run of download_images, it might already be clean.

    # Ensure .env is loaded (should be by the top-level load_dotenv)
    # The initial load_dotenv at the top of tools.py should cover this.
    asyncio.run(main_test_pdf_creation())
