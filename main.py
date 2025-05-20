import lmstudio as lms # Keep for ToolFunctionDef and other lmstudio specifics if used by tools
# from lmstudio import ToolFunctionDef # Now directly from lmstudio
from lmstudio.tools.functions_def import ToolFunctionDef # Corrected import for ToolFunctionDef
import time
import pick # Keep for LMStudioProvider's internal use if not fully abstracted
import requests
import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
# from dotenv import load_dotenv # Now also set_key
from dotenv import load_dotenv, set_key
import os # Added

# Import provider classes
from lmstudio_provider import LMStudioProvider
from openai_provider import OpenAIProvider
from llm_provider_interface import LLMProviderInterface

# print_fragment is removed as providers will handle their own streaming if necessary,
# or the interface will be updated for a generic streaming callback.

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
        return(result.markdown)  # Return clean markdown content

# Synchronous wrapper for crawl to be used in ToolFunctionDef implementation
def crawl_sync(url: str):
    """
    Run the async crawl function in a synchronous context and return markdown.
    """
    return asyncio.run(crawl(url))

tool_crawl_link = ToolFunctionDef(
  name="Crawl_a_webpage", # OpenAI compatible name
  description="Crawls a webpage and extracts markdown content. You need to give an exact URL with https://.",
  params_jsonschema={ # Changed to params_jsonschema
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "The URL to crawl, must include schema (e.g. https://)"},
    },
    "required": ["url"],
  },
  implementation=crawl_sync,
)

# Function to search Wikipedia API for sites
def wikipedia_search(query: str, limit: int = 25): # Added type hints for clarity
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
    # Return a list of titles, not just the first one.
    return [r['title'] for r in results if 'title' in r]

tool_search_pages = ToolFunctionDef(
  name="Wikipedia_Search_matching_pages", # OpenAI compatible name
  description="Searches Wikipedia pages for a given query with the total number of pagenames to return set with limit. limit is capped at 500 and has a default value of 25.",
  params_jsonschema={ # Changed to params_jsonschema
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query"},
        "limit": {"type": "integer", "description": "Maximum number of page titles to return", "default": 25},
    },
    "required": ["query"],
  },
  implementation=wikipedia_search,
)

def wikipedia(page: str): # Added type hint
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
        return "No extract found for the given page." # Consistent return type

tool_get_page_info = ToolFunctionDef(
  name="Wikipedia_fetches_a_page", # OpenAI compatible name
  description="Gets information from a Wikipedia page for a given exact title. The title must be exact.",
  params_jsonschema={ # Changed to params_jsonschema
    "type": "object",
    "properties": {
        "page": {"type": "string", "description": "The exact title of the Wikipedia page"},
    },
    "required": ["page"],
  },
  implementation=wikipedia,
)

# The old load_model() function is removed.

def select_and_load_provider() -> LLMProviderInterface:
    """
    Allows the user to select an LLM provider (LM Studio or OpenAI),
    configure it, and load a model.
    """
    load_dotenv()  # Load existing .env file if present

    while True:
        print("\nSelect LLM Provider:")
        choice = input("(1) LM Studio (local)\n(2) OpenAI API (remote)\nEnter choice (1 or 2), or 'q' to quit: ").strip().lower()

        if choice == 'q':
            return None

        if choice == "1":
            try:
                provider = LMStudioProvider()
                # load_model in LMStudioProvider handles its own model selection and error handling
                provider.load_model() 
                if provider.model is None: # LMStudioProvider.load_model should indicate failure by setting self.model to None
                    # Error message already printed by LMStudioProvider
                    print("LM Studio model loading failed. Please try again or select a different provider.")
                    continue # Re-prompt for provider selection
                print(f"Model loaded successfully via LMStudioProvider: {provider.get_model_info().get('identifier', 'N/A')}")
                return provider
            except Exception as e:
                print(f"Error loading provider LM Studio: {e}")
                # Optionally, loop back to provider selection or exit
                # For now, let's allow re-selection
                continue
        
        elif choice == "2":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                while True:
                    api_key_input = input("Enter your OpenAI API Key (or press Enter to skip if already set via other means, 'q' to go back): ").strip()
                    if not api_key_input:
                        api_key = os.getenv("OPENAI_API_KEY") # Re-check env var in case it was set externally
                        if not api_key:
                            print("OpenAI API Key is required to proceed with this provider.")
                            # Decide to re-prompt for key or go back to provider selection
                            # For now, re-prompt for key:
                            continue 
                        else:
                            break # API key found in env
                    elif api_key_input.lower() == 'q':
                        break # Go back to provider selection
                    
                    save_key_choice = input("Save API Key to .env file for future sessions? (y/N): ").lower()
                    if save_key_choice == 'y':
                        set_key(".env", "OPENAI_API_KEY", api_key_input)
                        print("API Key saved to .env file.")
                    os.environ["OPENAI_API_KEY"] = api_key_input # Set for current session
                    api_key = api_key_input
                    break 
            if not api_key and choice == '2': # If 'q' was pressed for API key input
                continue # Go back to provider selection

            base_url_input = input("Enter API Base URL (leave blank for default OpenAI 'https://api.openai.com/v1', 'q' to go back): ").strip()
            if base_url_input.lower() == 'q':
                continue
            base_url = base_url_input if base_url_input else None # Use None for default

            model_name_input = ""
            while not model_name_input:
                model_name_input = input("Enter OpenAI Model Identifier (e.g., gpt-4-turbo-preview, 'q' to go back): ").strip()
                if model_name_input.lower() == 'q':
                    break
                if not model_name_input:
                    print("OpenAI Model Identifier is required.")
            if model_name_input.lower() == 'q':
                continue # Go back to provider selection

            try:
                provider = OpenAIProvider(api_key=api_key, base_url=base_url) # Error handling for API key in __init__
                provider.load_model(model_identifier=model_name_input) # Error handling for model_id in load_model
                print(f"Model '{model_name_input}' set successfully for OpenAIProvider.")
                # Verification of model happens in load_model, message printed there
                return provider
            except ValueError as ve: # Catch specific errors from provider
                print(f"Configuration error for OpenAI: {ve}")
                # Allow user to re-enter details or select new provider
                continue
            except Exception as e:
                print(f"Error initializing OpenAI provider or loading model: {e}")
                # Allow user to re-enter details or select new provider
                continue
        else:
            print("Invalid selection, please try again.")
            # Loop continues for provider selection

# This logic will be handled by the provider or within the main loop if needed.

def check_context_fit(provider: LLMProviderInterface, messages: list, max_ratio_threshold=0.75) -> bool:
    try:
        # Attempt to use apply_prompt_template then tokenize
        # This is a rough estimate as tokenization can vary
        # Note: apply_prompt_template for OpenAI returns JSON string, for LMStudio it's the actual prompt.
        # Tokenization needs to be robust to this. OpenAIProvider.tokenize currently returns [].
        # This check will be more effective with actual tokenization from providers.
        
        prompt_for_tokenization = ""
        if isinstance(provider, OpenAIProvider):
            # For OpenAI, to get a token count estimate for messages,
            # it's better to join content, as apply_prompt_template returns JSON of messages.
            # Or use tiktoken if available. For now, simple concatenation for a rough idea.
            prompt_for_tokenization = " ".join(msg.get("content", "") for msg in messages)
        else: # For LMStudioProvider and potentially others
            prompt_for_tokenization = provider.apply_prompt_template(messages)

        token_count = len(provider.tokenize(prompt_for_tokenization))
        context_length = provider.get_context_length()

        if context_length > 0 and token_count > 0: # Ensure valid numbers
            ratio = token_count / context_length
            print(f"Context utilization estimate: {token_count}/{context_length} tokens ({ratio*100:.2f}%)")
            if ratio >= max_ratio_threshold:
                print("Warning: Conversation history might be approaching the model's context limit.")
            return True # Return true to proceed, warning is informational
        elif context_length == 0 and token_count > 0 : # Model has context length 0 (e.g. LMStudio not fully loaded)
            print(f"Warning: Model context length is 0. Cannot check context fit. Token count: {token_count}")
            return False # Cannot fit if context length is 0
        return True # Default to true if info not available or token_count is 0
    except NotImplementedError:
        print("Tokenization/context length not fully supported by provider, skipping context fit check.")
        return True # Assume it fits if provider doesn't support
    except Exception as e:
        print(f"Error in context fit check: {e}")
        return True # Default to true on other errors

# Main function
def main():
    provider = select_and_load_provider()

    if provider is None:
        print("Provider setup failed. Exiting.")
        return

    print(f"\nUsing provider: {provider.__class__.__name__}")
    model_info = provider.get_model_info()
    print(f"Model loaded: {model_info.get('identifier', 'N/A')}")
    
    # Ensure context_length is displayed correctly, even if it's 0 or not available
    ctx_len = model_info.get('context_length', 'N/A')
    if ctx_len == -1 : ctx_len = "Not available or N/A"
    print(f"Model context length: {ctx_len}")


    messages = [{"role": "system", "content": "You are a task focused AI researcher. Be concise."}]
    available_tools = [tool_search_pages, tool_get_page_info, tool_crawl_link]

    while True:
        try:
            user_input = input("You (leave blank to exit): ")
        except EOFError:
            print()
            break
        if not user_input:
            break
        
        messages.append({"role": "user", "content": user_input})
        
        if not check_context_fit(provider, messages):
            # Optionally, implement context management here (e.g., summarize, truncate)
            # For now, just warn and proceed, or potentially break/return if critical
            print("Critical Warning: Context length might be exceeded or model not fully loaded. Consider restarting or a shorter conversation.")
            # continue # Or break, depending on desired strictness

        print("Bot: ", end="", flush=True) # Print "Bot: " before the response starts streaming or arrives
        
        try:
            # Pass the print_fragment callback only if it's an LMStudioProvider instance
            # and its chat_completion method is expecting it.
            # Based on current LMStudioProvider, it handles printing internally via _print_fragment
            # when on_prediction_fragment is passed to model.act().
            # So, we don't need to pass main.print_fragment here directly.
            # The LMStudioProvider's chat_completion should internally use its _print_fragment.
            
            assistant_response_content = provider.chat_completion(
                messages=messages,
                tools=available_tools
                # temperature=0.7 # Example of another potential kwarg
            )
            
            # If chat_completion returns the full string (as it does now per interface)
            # and if the provider (like LMStudioProvider) already printed fragments,
            # printing here again would duplicate.
            # For OpenAIProvider, it returns the full string, so we print it.
            # This needs careful handling based on provider behavior.

            if not isinstance(provider, LMStudioProvider): # LMStudioProvider prints fragments itself
                 print(assistant_response_content) # Print full response for others
            else:
                 print() # LMStudioProvider already printed, just add a newline if needed after fragments.

            if assistant_response_content: # Ensure there's content to append
                messages.append({"role": "assistant", "content": assistant_response_content})
            # An empty response isn't necessarily a warning if the model chose not to speak
            # else:
            #     print("Warning: Assistant response was empty.")

        except NotImplementedError as nie:
            print(f"\nAn error occurred: Functionality not implemented in the provider: {nie}")
            # Optionally add a placeholder error message to chat history
            # messages.append({"role": "assistant", "content": f"[Error: Feature not implemented - {nie}]"})
        except Exception as e:
            # User-friendly message for general errors during chat completion
            print(f"\nAn error occurred during chat completion: {e}")
            # Optionally add a placeholder error message to chat history
            # messages.append({"role": "assistant", "content": "[Error communicating with LLM. Please try again.]"})
        
        # print() # Ensure a newline after the bot's full response or error message.
        # The current printing logic (Bot: prefix, then response, then newline by assistant's print or our print())
        # should be okay.

# Start the program
if __name__ == "__main__":
    main()