import lmstudio as lms
from model_loader import load_model, does_chat_fit_in_context
from web_crawler import crawl_sync
from wikipedia_tools import wikipedia_search, wikipedia
from lmstudio import ToolFunctionDef

def print_fragment(fragment, round_index=0):
    """
    Prints the content of a prediction fragment to standard output without a newline.
    
    Args:
        fragment: An object containing a 'content' attribute to be printed.
        round_index: Unused parameter included for interface compatibility.
    """
    print(fragment.content, end="", flush=True)

tool_crawl_link = ToolFunctionDef(
  name="Crawl a webpage",
  description="Crawls a webpage and extracts markdown content. You need to give an exact URL with https://.",
  parameters={
    "url": str,
  },
  implementation=crawl_sync,
)

tool_search_pages = ToolFunctionDef(
  name="Wikipedia Search matching pages",
  description="Searches Wikipedia pages for a given query with the total number of pagenames to return set with limit. limit is capped at 500 and has a default value of 25.",
  parameters={
    "query": str,
    "limit": int,
  },
  implementation=wikipedia_search,
)

tool_get_page_info = ToolFunctionDef(
  name="Wikipedia fetches a page",
  description="Gets information from a Wikipedia page for a given exact title. The title must be exact.",
  parameters={
    "page": str,
  },
  implementation=wikipedia,
)

def main():
    """
    Runs the interactive command-line chat interface for the AI assistant.
    
    Continuously prompts the user for input, manages the chat session, and streams model responses with integrated web and Wikipedia tools. Exits on blank input or EOF.
    """
    load_model()
    model = lms.llm()
    chat = lms.Chat("You are a task focused AI researcher")
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

if __name__ == "__main__":
    main()
