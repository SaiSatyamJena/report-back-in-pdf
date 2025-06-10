if __name__ == "__main__":
    import asyncio

    async def main_test():
        # ... (existing test for search_web_for_topic can remain) ...

        # Test the find_image_urls_for_topic function
        image_test_topic = "beautiful landscapes"
        print(f"\n--- Testing find_image_urls_for_topic with topic: '{image_test_topic}' ---")
        image_urls_data = await find_image_urls_for_topic(image_test_topic, num_images=2)
        
        if image_urls_data:
            print("\n--- Found Image URLs & Titles ---")
            for item in image_urls_data:
                print(f"  Title: {item['title']}")
                print(f"  URL: {item['url']}")
        else:
            print("  No image URLs found.")
        print("--- End of Image Test ---")

    # Make sure load_dotenv is called if you run this standalone and it's not at the top level
    # dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # Recalculate if needed
    # load_dotenv(dotenv_path=dotenv_path)
    asyncio.run(main_test())