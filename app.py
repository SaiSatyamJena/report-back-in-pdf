import gradio as gr
import asyncio
import os
from pathlib import Path

# MCP Client Imports
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import (
    CallToolResult,
    TextContent,
)  # Import CallToolResult for type checking

# Configuration for the MCP Server
MCP_SERVER_BASE_URL = "http://localhost:8000"
MCP_SERVER_MCP_ENDPOINT = f"{MCP_SERVER_BASE_URL}/mcp"


# --- Core Logic Function ---
async def get_report_from_mcp_server(topic: str, num_images: int):
    """
    Contacts the MCP server to request a report.
    Returns a tuple: (status_message_for_ui: str, pdf_file_path_or_none: str or None)
    """
    print(
        f"[App/MCP Client] Requesting report for topic: '{topic}', num_images: {num_images}"
    )

    status_message = "Initiating contact with server..."
    pdf_file_path_for_download = None

    try:
        print(
            f"[App/MCP Client] Connecting to MCP server at {MCP_SERVER_MCP_ENDPOINT}..."
        )
        async with streamablehttp_client(MCP_SERVER_MCP_ENDPOINT) as (
            read_stream,
            write_stream,
            _,
        ):
            print(
                "[App/MCP Client] Connection established. Creating MCP client session..."
            )
            async with ClientSession(read_stream, write_stream) as session:
                print(
                    "[App/MCP Client] MCP client session created. Initializing session..."
                )
                await session.initialize()
                print(
                    "[App/MCP Client] Session initialized. Calling 'generate_pdf_report_on_topic' tool..."
                )

                tool_arguments = {
                    "topic": topic,
                    "num_images": int(num_images),
                }

                # This is the raw result from the server tool
                mcp_tool_result_obj: CallToolResult = await session.call_tool(
                    name="generate_pdf_report_on_topic", arguments=tool_arguments
                )

                print(
                    f"[App/MCP Client] Tool call completed. Raw MCP object from server: {mcp_tool_result_obj} (Type: {type(mcp_tool_result_obj)})"
                )

                # --- NEW LOGIC TO HANDLE CallToolResult ---
                actual_content_str = None
                if mcp_tool_result_obj.isError:
                    # Try to extract error message if available
                    if mcp_tool_result_obj.content and isinstance(
                        mcp_tool_result_obj.content[0], TextContent
                    ):
                        actual_content_str = (
                            "Error: " + mcp_tool_result_obj.content[0].text
                        )  # Prepend "Error:" for consistency
                    else:
                        actual_content_str = "Error: Server indicated an error, but no specific message was found in content."
                    print(
                        f"[App/MCP Client] MCP Result isError=True. Content: {actual_content_str}"
                    )

                elif mcp_tool_result_obj.content and isinstance(
                    mcp_tool_result_obj.content[0], TextContent
                ):
                    actual_content_str = mcp_tool_result_obj.content[0].text
                    print(
                        f"[App/MCP Client] MCP Result isError=False. Extracted text content: '{actual_content_str}'"
                    )
                else:
                    status_message = (
                        f"Critical: Server returned a non-error MCP result, but content was not in the expected TextContent format. "
                        f"Got content: '{mcp_tool_result_obj.content}'. Check console."
                    )
                    # actual_content_str remains None, handled below

                # --- END NEW LOGIC ---

                if actual_content_str is not None:
                    if actual_content_str.lower().startswith("error:"):
                        status_message = f"Server Feedback: {actual_content_str}"  # Server itself provided an error string
                        pdf_file_path_for_download = None
                    else:
                        # Assume it's a file path if not an error string
                        try:
                            p = Path(actual_content_str)
                            if p.is_file() and p.suffix.lower() == ".pdf":
                                status_message = f"Report generated successfully! Path: {actual_content_str}"
                                pdf_file_path_for_download = str(p.resolve())
                            else:
                                status_message = f"Server returned an unexpected file or path: '{actual_content_str}'. Ensure it's a PDF."
                                pdf_file_path_for_download = None
                        except Exception as path_e:
                            status_message = f"Server returned a string, but it's not a valid path ('{actual_content_str}'). Error: {path_e}"
                            pdf_file_path_for_download = None
                elif not status_message.startswith(
                    "Critical:"
                ):  # If status_message wasn't set by the critical error above
                    status_message = (
                        "Critical: Could not extract a usable string (path or error) from the server's response. "
                        "Check console for details on the raw MCP object."
                    )
                    pdf_file_path_for_download = None

    except ConnectionRefusedError:
        error_msg = f"Connection Error: Could not connect to the MCP Server at {MCP_SERVER_BASE_URL}. Is the server (server.py) running?"
        print(f"[App/MCP Client] {error_msg}")
        status_message = error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred in app while communicating with MCP: {str(e)}"
        print(f"[App/MCP Client] {error_msg} (Type: {type(e).__name__})")
        status_message = error_msg

    return status_message, pdf_file_path_for_download


# --- Gradio UI Definition ---

# Minimalist CSS
custom_css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #ffffff; color: #000000; }
.gradio-container { max-width: none !important; padding: 20px; } /* Full width */
.status-box { padding: 12px; border-radius: 6px; margin-top: 15px; font-size: 1em; line-height: 1.6; border: 1px solid #ccc;}
.status-success { background-color: #e6fffa; border-color: #b2f5ea; color: #2c7a7b; }
.status-error   { background-color: #fff5f5; border-color: #fecaca; color: #c53030; }
.status-processing{ background-color: #ebf8ff; border-color: #bee3f8; color: #2c5282; }
.status-info    { background-color: #fffff0; border-color: #fefcbf; color: #744210; }
.footer-text { font-size: 0.85em; color: #718096; text-align: center; margin-top: 30px; padding-top: 15px; border-top: 1px solid #e2e8f0; }
h1, h3 {color: #2d3748;}
.gr-button-primary { background-color: #4299e1 !important; color: white !important; } /* Example primary button color */
"""

with gr.Blocks(css=custom_css, title="AI PDF Report Generator") as app_ui:
    gr.Markdown(
        "<h1 style='text-align: center; margin-bottom: 5px;'>🤖 AI PDF Report Generator</h1>"
        "<p style='text-align: center; color: #4a5568; margin-bottom: 25px;'>Enter a topic and number of images to generate your PDF report.</p>"
    )

    topic_input = gr.Textbox(
        label="Report Topic",
        placeholder="e.g., The Future of Space Exploration",
        info="Be specific for better results.",
        # lines=1 # Default is single line
    )
    num_images_input = gr.Slider(
        minimum=0,
        maximum=5,
        step=1,
        value=2,
        label="Number of Images (0-5)",
        info="How many images should the report include?",
    )
    submit_button = gr.Button(
        "🚀 Generate Report", variant="primary", size="lg"
    )  # Primary button

    status_markdown = gr.Markdown(  # Moved status directly below inputs
        "<div class='status-box status-info'>ℹ️ Ready when you are!</div>"
    )
    pdf_output_file = gr.File(  # Moved download directly below status
        label="📥 Download Your Report Here:",
        interactive=False,
    )

    async def handle_submit_action(topic_str: str, num_img_int: int):
        if not topic_str.strip():
            error_html = "<div class='status-box status-error'><strong>⚠️ Input Error:</strong> Please enter a topic for the report.</div>"
            yield error_html, None
            return

        processing_html = f"<div class='status-box status-processing'>⚙️ <strong>Processing:</strong> Generating report for '<em>{topic_str}</em>' with {num_img_int} image(s). This may take a moment...</div>"
        yield processing_html, None

        status_text_from_mcp, generated_pdf_path = await get_report_from_mcp_server(
            topic_str, num_img_int
        )

        final_status_html = ""
        output_file_for_ui = None

        if generated_pdf_path and Path(generated_pdf_path).is_file():
            file_name = Path(generated_pdf_path).name
            final_status_html = f"<div class='status-box status-success'><strong>✅ Success!</strong> Report '<code>{file_name}</code>' is ready. You can download it below.</div>"
            output_file_for_ui = generated_pdf_path
        elif "Error:" in status_text_from_mcp or "Critical:" in status_text_from_mcp:
            final_status_html = f"<div class='status-box status-error'><strong>❌ {status_text_from_mcp.split(':', 1)[0]}:</strong> {status_text_from_mcp.split(':', 1)[-1].strip()}</div>"
        else:
            final_status_html = f"<div class='status-box status-info'><strong>ℹ️ Status:</strong> {status_text_from_mcp}</div>"

        yield final_status_html, output_file_for_ui

    submit_button.click(
        fn=handle_submit_action,
        inputs=[topic_input, num_images_input],
        outputs=[status_markdown, pdf_output_file],
    )

    gr.Examples(
        examples=[
            ["The History of the Internet", 2],
            ["Sustainable Energy Solutions", 3],
            ["The Wonders of Ancient Egypt", 1],
            ["Mars Colonization Challenges", 0],
        ],
        inputs=[topic_input, num_images_input],
        label="💡 Example Topics (click to try)",
        # outputs=[status_markdown, pdf_output_file],
        # fn=handle_submit_action
    )

    gr.Markdown(
        "Powered by an MCP AI Agent & Gradio. Ensure the MCP server (`server.py`) is running.",
        elem_classes="footer-text",
    )

if __name__ == "__main__":
    print("[Gradio App] Launching Gradio interface...")
    app_ui.launch(share=False, server_name="0.0.0.0", server_port=7860)
    print(
        "[Gradio App] Gradio interface has been launched. Check your browser or the provided links."
    )
