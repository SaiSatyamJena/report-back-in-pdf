# D:\repos\report-back-in-pdf\mcp_reporter_app\server.py

import asyncio
from mcp.server.fastmcp import FastMCP  # Use FastMCP for simpler server setup

# If FastMCP is not found directly under mcp, it might be:
# from mcp.server.fastmcp import FastMCP
from pathlib import Path

# Import the tools from our tools_clean.py file
# The '.' means "from the current package" (mcp_reporter_app)
from .tools_clean import (
    search_web_for_topic,
    find_image_urls_for_topic,
    download_images,
    generate_report_text_content,
    create_report_pdf,
    MAX_PDF_PAGES,  # We might want this for the orchestrator's docstring or logic
)

# --- MCP Server Setup ---
# Create an MCP server instance. The name is for identification.
# Based on the docs, FastMCP is the high-level server.
try:
    # Check if FastMCP can be imported directly from mcp
    # This might vary slightly based on the exact version of the mcp package installed by "mcp[cli]"
    # If `from mcp import FastMCP` fails, try `from mcp.server.fastmcp import FastMCP`
    # and adjust the import statement at the top of this file.
    mcp_server_instance = FastMCP(
        name="PDFReportGeneratorServer",
        description="A server that generates PDF reports on given topics.",
        # You can add version, author, etc. if desired
        # version="0.1.0",
        # author="Your Name"
    )
    print("[MCP Server] FastMCP instance created.")
except ImportError:
    print(
        "ImportError: Could not import FastMCP directly from 'mcp'. Trying 'mcp.server.fastmcp'..."
    )
    from mcp.server.fastmcp import FastMCP  # Fallback import

    mcp_server_instance = FastMCP(
        name="PDFReportGeneratorServer",
        description="A server that generates PDF reports on given topics.",
    )
    print("[MCP Server] FastMCP instance created using mcp.server.fastmcp.")
except Exception as e:
    print(f"Failed to initialize FastMCP server: {e}")
    mcp_server_instance = None


# --- Orchestrator Tool Definition ---
# This is the main function that our client (Gradio app) will call.
# It orchestrates the calls to the individual tools.
async def generate_full_pdf_report(topic: str, num_images: int = 2) -> str:
    """
    Generates a full PDF report on a given topic.
    This involves searching the web, finding and downloading images,
    generating text content with OpenAI, and compiling it all into a PDF.
    The report will be a maximum of approx. MAX_PDF_PAGES pages.

    Args:
        topic (str): The topic for the report.
        num_images (int): The desired number of images to include (0-5 recommended).

    Returns:
        str: The absolute local file path to the generated PDF, or an error message string.
    """
    print(
        f"\n[Orchestrator] Received request to generate report for topic: '{topic}', num_images: {num_images}"
    )

    # Step 1: Search web for topic content
    print("[Orchestrator] Step 1: Searching web...")
    web_summary = await search_web_for_topic(topic, num_results=3)
    if (
        "Error" in web_summary or "No results found" in web_summary.split("\n", 1)[-1]
    ):  # Check after initial line
        print(f"[Orchestrator] Web search failed or found no results: {web_summary}")
        return f"Error: Web search failed or found no results for '{topic}'."
    print("[Orchestrator] Web search successful.")

    # Step 2: Find image URLs
    downloaded_images_final_info = []  # Default to empty list
    if num_images > 0:
        print(f"[Orchestrator] Step 2: Finding {num_images} image URLs...")
        image_urls_data = await find_image_urls_for_topic(topic, num_images=num_images)
        if not image_urls_data:
            print(
                "[Orchestrator] No image URLs found. Report will be text-only or have fewer images."
            )
        else:
            print(f"[Orchestrator] Found {len(image_urls_data)} image URLs.")

            # Step 3: Download images
            print("[Orchestrator] Step 3: Downloading images...")
            # Pass clean_temp_dir=True to ensure a fresh download set for each report generation
            downloaded_images_final_info = await download_images(image_urls_data)
            if not downloaded_images_final_info:
                print(
                    "[Orchestrator] Failed to download any images. Report will be text-only or have fewer images."
                )
            else:
                print(
                    f"[Orchestrator] Successfully downloaded {len(downloaded_images_final_info)} images."
                )
    else:
        print("[Orchestrator] Skipping image search and download as num_images is 0.")

    # Step 4: Generate report text content using OpenAI
    print("[Orchestrator] Step 4: Generating report text content with OpenAI...")
    report_text_content = await generate_report_text_content(
        topic=topic,
        web_search_summary=web_summary,
        downloaded_images_info=downloaded_images_final_info,  # Pass info of actually downloaded images
        max_pages_target=MAX_PDF_PAGES,
    )
    if "error" in report_text_content or "Error" in report_text_content.get(
        "title", ""
    ):
        error_detail = report_text_content.get(
            "error", report_text_content.get("introduction", "Unknown OpenAI error")
        )
        print(f"[Orchestrator] OpenAI content generation failed: {error_detail}")
        return f"Error: OpenAI content generation failed for '{topic}'. Detail: {error_detail}"
    print("[Orchestrator] Report text content generation successful.")

    # Step 5: Create the PDF
    print("[Orchestrator] Step 5: Creating PDF report...")
    pdf_path_or_error = await create_report_pdf(
        report_content=report_text_content,
        downloaded_images_info=downloaded_images_final_info,
    )

    if isinstance(pdf_path_or_error, str) and "Error" in pdf_path_or_error:
        print(f"[Orchestrator] PDF creation failed: {pdf_path_or_error}")
        return f"Error: PDF creation failed for '{topic}'. Detail: {pdf_path_or_error}"
    elif isinstance(pdf_path_or_error, str) and Path(pdf_path_or_error).exists():
        print(f"[Orchestrator] PDF report successfully generated: {pdf_path_or_error}")
        return pdf_path_or_error  # Return the path
    else:
        print(
            f"[Orchestrator] PDF creation returned an unexpected result: {pdf_path_or_error}"
        )
        return "Error: PDF creation returned an unexpected or invalid result."


# --- Register the Orchestrator as an MCP Tool ---
if mcp_server_instance:
    # The @mcp_server_instance.tool() decorator registers the function.
    # We can also define it explicitly if the decorator syntax is tricky with existing async defs
    # or if we want more control, but the decorator is standard for FastMCP.

    # Let's try defining and adding it more explicitly to avoid potential decorator issues
    # with complex async functions, though decorator should work.

    # Method 1: Using the decorator (standard FastMCP way)
    @mcp_server_instance.tool(
        name="generate_pdf_report_on_topic",  # This is how the client will call it
        description="Generates a multi-page PDF report on a specified topic, including web research, image integration, and AI-generated content.",
        # We can also add input_schema for typed arguments if needed, but not critical for MVP
    )
    async def decorated_generate_full_pdf_report(
        topic: str, num_images: int = 2
    ) -> str:
        # This wrapper simply calls our main orchestrator logic
        return await generate_full_pdf_report(topic, num_images)

    print(
        f"[MCP Server] Tool '{decorated_generate_full_pdf_report.__name__}' (exposed as 'generate_pdf_report_on_topic') registered."
    )

else:
    print("[MCP Server] MCP Server instance not created. Cannot register tools.")


# # --- Main Entry Point to Run the Server ---
# async def main():
#     if not mcp_server_instance:
#         print("Cannot start server because FastMCP instance failed to initialize.")
#         return

#     print("[MCP Server] Starting PDF Report Generator Server...")
#     print("  Available tools should be listed by the MCP framework upon connection.")
#     print(
#         "  Server will listen for MCP client connections (e.g., from Gradio app or MCP Inspector)."
#     )

#     # To run the FastMCP server, we call its run() method.
#     # By default, FastMCP with .run() will likely use Streamable HTTP transport
#     # listening on a default port (often 8000 or 8080, check MCP docs or output).
#     # You can specify transport and port: mcp_server_instance.run(transport="streamable-http", port=8008)
#     try:
#         await mcp_server_instance.run()  # Default transport and port
#     except Exception as e:
#         print(f"Error running MCP server: {e}")


if __name__ == "__main__":
    # Ensure environment variables are loaded if this script is run directly
    from dotenv import load_dotenv as load_dotenv_server

    env_path_server = Path(__file__).parent.parent / ".env"
    if env_path_server.exists():
        load_dotenv_server(dotenv_path=env_path_server)
        print(f"[Server Main] .env loaded from {env_path_server}")
    else:
        print(
            f"[Server Main] .env file not found at {env_path_server}. Relies on tools_clean.py loading."
        )

    if not mcp_server_instance:
        print("Cannot start server because FastMCP instance failed to initialize.")
    else:
        print("[MCP Server] Starting PDF Report Generator Server...")
        print(
            "  Available tools should be listed by the MCP framework upon connection."
        )
        print(
            "[MCP Server] Attempting to start server with Streamable HTTP transport..."
        )
        try:
            # Explicitly tell FastMCP to use streamable-http and specify a port
            # This is the crucial change to make it an HTTP server.
            mcp_server_instance.run(transport="streamable-http")

        except KeyboardInterrupt:
            # This will catch Ctrl+C if mcp_server_instance.run() doesn't handle it
            # before it propagates, or if the interrupt happens during setup by run().
            print(
                "\n[MCP Server] KeyboardInterrupt received by __main__. Server is stopping."
            )
        except SystemExit:
            # Uvicorn (used by FastMCP for HTTP) often raises SystemExit on Ctrl+C.
            print(
                "\n[MCP Server] SystemExit received (likely Ctrl+C via Uvicorn). Server is stopping."
            )
        except Exception as e:
            print(f"[MCP Server] An error occurred while trying to run the server: {e}")
        finally:
            # This message will print after the server stops, regardless of how.
            print("[MCP Server] Server execution in __main__ has finished.")
