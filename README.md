# 🤖 AI PDF Report Generator (via MCP) 📄

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app)
[![OpenAI API](https://img.shields.io/badge/AI-OpenAI%20API-4EAA8E)](https://openai.com/api/)

Transform simple topics into well-structured PDF reports! This project leverages the **Model Context Protocol (MCP)** for a robust client-server architecture, **OpenAI's GPT models** for intelligent content generation, and **Gradio** for an intuitive web interface.

Enter a topic, specify the number of images, and watch as the system researches, writes, and compiles a downloadable PDF report for you.

---

## ✨ Core Features

*   **Dynamic Topic Reporting:** Generate reports on virtually any subject.
*   **Automated Web Research:** Utilizes DuckDuckGo to gather up-to-date information.
*   **AI-Powered Content Synthesis:** Employs OpenAI's GPT-3.5-turbo to craft:
    *   Engaging Titles
    *   Informative Introductions
    *   Structured Main Sections with Headings
    *   Concise Conclusions
*   **Intelligent Image Integration:**
    *   Searches for relevant images related to the topic.
    *   Downloads and embeds selected images into the PDF.
    *   Handles potential image download/processing errors gracefully.
*   **PDF Generation:** Produces a clean, multi-page PDF document using FPDF2 (up to 10 pages).
*   **MCP Client-Server Architecture:**
    *   **MCP Server (`server.py`):** An HTTP server built with `FastMCP`. It exposes a primary "orchestrator" tool (`generate_pdf_report_on_topic`) that manages the entire report generation workflow.
    *   **MCP Client (`app.py`):** A user-friendly Gradio web UI that communicates with the MCP Server. It sends requests to the server's tool and processes the results.
    *   **Specialized Tools (`tools_clean.py`):** A collection of Python functions performing discrete tasks (web search, image handling, OpenAI interaction, PDF creation), called by the server's orchestrator.
*   **Asynchronous Operations:** Built with `asyncio` for efficient, non-blocking I/O operations, especially during web requests and API calls.
*   **Interactive & Responsive UI:** A clean, minimalist Gradio interface that's easy to use.

---

## 🚀 Live Demo

![AI PDF Report Generator Demo](./mcp-reporter.gif)

---

## 🛠️ Tech Stack

*   **Programming Language:** Python 3.9+
*   **AI & Language Models:**
    *   OpenAI API (`openai` library, GPT-3.5-turbo)
*   **Client-Server Communication (MCP):**
    *   `mcp` library:
        *   `FastMCP`: For building the MCP Server (`server.py`).
        *   `ClientSession` & `streamablehttp_client`: For the MCP Client (`app.py`) to communicate with the server.
*   **Web UI Framework:**
    *   Gradio (`gradio`)
*   **PDF Generation:**
    *   FPDF2 (`fpdf2`)
*   **Web Scraping & Search:**
    *   DuckDuckGo Search (`duckduckgo-search`)
*   **Image Handling:**
    *   Pillow (`Pillow`) for image processing.
    *   Requests (`requests`) for downloading images.
*   **Environment Management:**
    *   `python-dotenv` for API key management.
*   **Concurrency:**
    *   `asyncio` for asynchronous programming.

---

## ⚙️ Setup & Installation

Get the AI PDF Report Generator up and running on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/SaiSatyamJena/report-back-in-pdf.git
    cd report-back-in-pdf
    ```
   

2.  **Create and Activate a Python Virtual Environment:**
    (This keeps project dependencies isolated)
    ```bash
    python -m venv .venv
    ```
    *   Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
    *   Windows (Git Bash/CMD): `source .venv/Scripts/activate`
    *   macOS/Linux: `source .venv/bin/activate`

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    *   In the project's root directory (`report-back-in-pdf/`), create a file named `.env`.
    *   Add your OpenAI API key to this `.env` file:
        ```env
        OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY_HERE"
        ```
    *   **Important:** Ensure the `.env` file is listed in your `.gitignore` file to prevent accidentally committing your secret API key.

---

## ▶️ Running the Application

The application consists of two main parts: the MCP Server and the Gradio UI (MCP Client). You'll need to run them in separate terminal windows.

1.  **Start the MCP Server (Terminal 1):**
    *   Make sure your virtual environment is activated.
    *   Navigate to the project root directory.
    *   Execute:
        ```bash
        python -m mcp_reporter_app.server
        ```
    *   You should see log messages indicating the server has started, typically mentioning "Uvicorn running on http://127.0.0.1:8000". This server listens for requests from MCP clients.

2.  **Start the Gradio Web UI / MCP Client (Terminal 2):**
    *   Activate your virtual environment in this new terminal.
    *   Navigate to the project root directory.
    *   Execute:
        ```bash
        python app.py
        ```
    *   The console will output a local URL, usually `http://0.0.0.0:7860` or `http://127.0.0.1:7860`. This is where you'll interact with the application.

3.  **Access the Application:**
    *   Open your preferred web browser and navigate to the URL provided by the Gradio app (e.g., `http://127.0.0.1:7860`).
    *   You should now see the AI PDF Report Generator interface. Enter a topic, select image count, and generate your report!

---

## 📖 How It Works - The Flow

The application follows a clear client-server model facilitated by the Model Context Protocol (MCP):

1.  **User Interaction (Gradio UI - `app.py` as MCP Client):**
    *   The user provides a topic and the desired number of images through the Gradio web interface.
    *   `app.py` acts as an **MCP Client**. When the "Generate Report" button is clicked, it establishes a connection to the **MCP Server** (`server.py`) running on `http://localhost:8000`.
    *   The client then invokes a specific tool exposed by the server, named `generate_pdf_report_on_topic`, passing the user's topic and image count as arguments.

2.  **MCP Server Orchestration (`server.py`):**
    *   The **MCP Server**, built using `FastMCP`, receives the tool invocation request.
    *   The `generate_pdf_report_on_topic` tool defined in `server.py` (which internally calls the `generate_full_pdf_report` async function) acts as an **orchestrator**. It doesn't perform all tasks itself but coordinates a sequence of calls to specialized, more granular "worker" functions.

3.  **Execution of Specialized Tools (`tools_clean.py`):**
    *   The orchestrator function in `server.py` calls various helper functions (the actual "tools") located in `mcp_reporter_app/tools_clean.py`. This sequence typically involves:
        1.  `search_web_for_topic()`: Gathers textual information from the web using DuckDuckGo.
        2.  `find_image_urls_for_topic()`: Searches for relevant image URLs.
        3.  `download_images()`: Downloads the identified images to a temporary local directory.
        4.  `generate_report_text_content()`: Sends the topic, web summary, and image details to the OpenAI API. OpenAI then generates structured report content (title, introduction, sections, conclusion) in JSON format.
        5.  `create_report_pdf()`: Compiles the AI-generated text and downloaded images into a final PDF document using FPDF2.

4.  **Response to Client (via MCP):**
    *   Once all steps are complete (or if an error occurs), the orchestrator tool on the **MCP Server** prepares a response.
    *   This response is packaged into an MCP `CallToolResult` object.
        *   On success, this object contains the absolute file path of the generated PDF (as `TextContent`).
        *   On failure, it contains an error message (also as `TextContent`).
    *   The MCP Server sends this `CallToolResult` back to the **MCP Client** (`app.py`).

5.  **UI Update (Gradio UI - `app.py`):**
    *   The **MCP Client** (`app.py`) receives the `CallToolResult` from the server.
    *   It inspects the result:
        *   If it's a file path, it updates the Gradio UI status message to "Success!" and makes the PDF available for download through the `gr.File` component.
        *   If it's an error message, it displays the error in the status message area.

This architecture separates concerns: the Gradio UI handles user interaction and client-side logic, the MCP server orchestrates the complex task, and the individual tools in `tools_clean.py` focus on specific operations.

---

## 💡 Future Enhancements & Ideas

*   **Advanced PDF Styling:** Implement more sophisticated PDF templates, custom fonts, better image placement (e.g., text wrapping), and support for tables or charts.
*   **Interactive Image Selection:** Allow users to preview and select/deselect images before they are included in the report.
*   **Enhanced AI Capabilities:**
    *   Utilize multimodal models (e.g., GPT-4V) for better image-text relevance checking.
    *   Option to use image generation models (like DALL-E) for custom illustrations.
    *   Allow choice of different GPT models or fine-tuning for specific report styles.
*   **Improved Error Handling:** More granular error messages in the UI and robust retry mechanisms for network-dependent tasks.
*   **Caching:** Implement caching for web search results or OpenAI API responses for common topics to reduce latency and API costs.
*   **User Configuration:** More UI options for report length, style preferences, specific data sources to include/exclude.
*   **Addressing FPDF Unicode Issues:** Implement a Unicode-supporting font with FPDF2 to correctly handle a wider range of characters in image titles and content.
*   **Alternative Image APIs:** Integrate with services like Pexels or Unsplash (with API key management) for more reliable image sourcing.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements, new features, or find any bugs, please feel free to:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourAmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some YourAmazingFeature'`).
5.  Push to the branch (`git push origin feature/YourAmazingFeature`).
6.  Open a Pull Request.

Alternatively, you can open an issue with the tag "enhancement" or "bug".

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE.md` file for more details.
*(You'll need to create a LICENSE.md file in your repository and paste the MIT License text into it.)*



---

## 🙏 Acknowledgements

*   The **Model Context Protocol (MCP)** project for providing the SDK and a novel approach to building AI agent systems.
*   **OpenAI** for their powerful and accessible language models.
*   **Gradio** for making it incredibly easy to build interactive machine learning web UIs.
*   The developers and communities behind **FPDF2, Pillow, DuckDuckGo-Search, Requests, python-dotenv**, and all other open-source libraries that made this project possible.

## 📂 Project Structure
```text
report-back-in-pdf/
├── .venv/ # Python virtual environment (managed locally)
├── mcp_reporter_app/ # Main application package
│ ├── init.py
│ ├── server.py # MCP Server: Exposes tools, orchestrates report generation
│ └── tools_clean.py # Core worker tools: web search, image handling, OpenAI, PDF creation
│ └── temp_downloaded_images/ # Temp storage for images (auto-cleared, gitignored)
├── generated_reports/ # Output directory for final PDFs (gitignored)
├── .git/
├── .gitignore # Specifies intentionally untracked files
├── app.py # Gradio UI (Acts as MCP Client)
├── mcp-reporter.gif # Demo GIF of the application
├── README.md # This file!
├── requirements.txt # Project dependencies
└── .env # Local environment variables (API Keys - NOT COMMITTED)
