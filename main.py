import lmstudio as lms
from lmstudio import ToolFunctionDef
import time
import pick
import requests
import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

def print_fragment(fragment, round_index=0):
    # .act() supplies the round index as the second parameter
    # Setting a default value means the callback is also
    # compatible with .complete() and .respond().
    print(fragment.content, end="", flush=True)

# Function to crawl a webpage and extract markdown content
async def crawl(url):
    browser_config = BrowserConfig()  # Default browser configuration
    run_config = CrawlerRunConfig()   # Default crawl run configuration
    print("Crawling URL:", url)
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=url,
            config=run_config
        )
        return(result.markdown)  # Print clean markdown content

# Synchronous wrapper for crawl to be used in ToolFunctionDef implementation
def crawl_sync(url: str):
    """
    Run the async crawl function in a synchronous context and return markdown.
    """
    return asyncio.run(crawl(url))

tool_crawl_link = ToolFunctionDef(
  name="Crawl a webpage",
  description="Crawls a webpage and extracts markdown content. You need to give an exact URL with https://.",
  parameters={
    "url": str,
  },
  implementation=crawl_sync,
)

# Function to search Wikipedia API for sites
def wikipedia_search(query, limit=25):
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
    'action': 'query',
    'format': 'json',
    'list': 'search',
    'srsearch': query,
    'srlimit': limit,
    }
    print("Searching Wikipedia for:", query)
    response = requests.get(url, params=params)
    data = response.json()
    results = data['query']['search']
    for r in results:
        return(r['title'])

tool_search_pages = ToolFunctionDef(
  name="Wikipedia Search matching pages",
  description="Searches Wikipedia pages for a given query with the total number of pagenames to return set with limit. limit is capped at 500 and has a default value of 25.",
  parameters={
    "query": str,
    "limit": int,
  },
  implementation=wikipedia_search,
)

def wikipedia(page):
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'extracts',
        'explaintext': True,
        'titles': page
    }
    print("Fetching Wikipedia page:", page)
    response = requests.get(url, params=params)
    data = response.json()
    page_data = next(iter(data['query']['pages'].values()))
    if 'extract' in page_data:
        return(page_data['extract'])
    else:
        return("No extract found for the given page.")

tool_get_page_info = ToolFunctionDef(
  name="Wikipedia fetches a page",
  description="Gets information from a Wikipedia page for a given exact title. The title must be exact.",
  parameters={
    "page": str,
  },
  implementation=wikipedia,
)

def load_model():
    if lms.list_loaded_models() == []:
        print("No models loaded")
        title = 'Please choose a model to load:'
        options = [model.model_key for model in lms.list_downloaded_models("llm")]
        option, index = pick.pick(options, title)
        with lms.Client() as client:
            
            model = client.llm.load_new_instance(options[index], config={
                "contextLength": 32768,
            })
            return model
    else:
        with lms.Client() as client:
            model = client.llm.model()
            if model.get_info().trained_for_tool_use == False:
                print("Model is not trained for tool use. "
                      "Please be careful when using it.")
                if input("Unload model? (y/N)") == "y":
                    model.unload()
                    return load_model()
                else:
                    return model
            else:
                print(model.get_info().identifier, "is already loaded.")
                if input("Unload model? (y/N)") == "y":
                    model.unload()
                    return load_model()
                else:
                    return model

def does_chat_fit_in_context(model: lms.LLM, chat: lms.Chat):
    # Convert the conversation to a string using the prompt template.
    formatted = model.apply_prompt_template(chat)
    # Count the number of tokens in the string.
    token_count = len(model.tokenize(formatted))
    # Get the current loaded context length of the model
    context_length = model.get_context_length()
    return context_length / token_count

# Main function
def main():
    # Load or select a model into memory
    load_model()
    model = lms.llm()
    chat = lms.Chat("You are a task focused AI researcher")
    # Example prompt
    while True:
        try:
            user_input = input("You (leave blank to exit): ")
        except EOFError:
            print()
            break
        if not user_input:
            break
        chat.add_user_message(user_input)
        print(does_chat_fit_in_context(model, chat), "Bot: ", end="", flush=True)
        model.act(
            chat,
            [tool_search_pages, tool_get_page_info, tool_crawl_link],
            on_message=chat.append,
            on_prediction_fragment=print_fragment,
        )
        print()





# Start the program
if __name__ == "__main__":
    main()