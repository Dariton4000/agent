#!/usr/bin/env python3
"""
AI Research Assistant with Wikipedia and Web Crawling capabilities.

This module provides an interactive AI assistant that can search Wikipedia,
fetch Wikipedia pages, and crawl web pages for research purposes.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any

import lmstudio as lms
from lmstudio import ToolFunctionDef
import pick
import requests
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    """Set up logging configuration based on environment variables."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Get verbose setting from environment (default: false)
    verbose_logging = os.getenv('VERBOSE_LOGGING', 'false').lower() == 'true'
    
    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/agent_{timestamp}.log'
    
    # Configure handlers
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set console handler level based on verbose setting
    if verbose_logging:
        console_handler.setLevel(logging.INFO)
    else:
        # Only show warnings and errors in console when not verbose
        console_handler.setLevel(logging.WARNING)
    
    handlers = [file_handler, console_handler]
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    return logging.getLogger(__name__)

# Setup logging
logger = setup_logging()


class AIResearchAssistant:
    """Main class for the AI Research Assistant."""
    
    def __init__(self):
        self.model: Optional[lms.LLM] = None
        self.client: Optional[lms.Client] = None
        self.chat: Optional[lms.Chat] = None
        self.tools: Dict[str, Any] = {}
        logger.info("AIResearchAssistant initialized")

    def print_fragment(self, fragment, round_index: int = 0) -> None:
        """Print AI response fragments as they are generated."""
        print(fragment.content, end="", flush=True)

    async def crawl_webpage(self, url: str) -> str:
        """
        Crawl a webpage and extract markdown content.
        
        Args:
            url: The URL to crawl
            
        Returns:
            Markdown content of the webpage
            
        Raises:
            Exception: If crawling fails
        """
        try:
            browser_config = BrowserConfig()
            run_config = CrawlerRunConfig()
            logger.info(f"Crawling URL: {url}")
            
            # Cosmetic CLI message
            print(f"🌐 Crawling webpage: {url}")
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=run_config)
                # Handle different possible result attributes for different Crawl4AI versions
                if hasattr(result, 'markdown'):
                    return result.markdown
                elif hasattr(result, 'cleaned_html'):
                    return result.cleaned_html
                elif hasattr(result, 'text'):
                    return result.text
                else:
                    return str(result)
        except Exception as e:
            logger.error(f"Failed to crawl {url}: {e}")
            raise

    def crawl_sync(self, url: str) -> str:
        """Synchronous wrapper for crawl_webpage."""
        try:
            # Note: The cosmetic message is already shown in crawl_webpage()
            try:
                loop = asyncio.get_running_loop()
                result = loop.run_until_complete(self.crawl_webpage(url))
            except RuntimeError:
                # No running loop → safe to create one
                result = asyncio.run(self.crawl_webpage(url))
            # Display context usage after tool call
            self.display_context_usage()
            return result
        except Exception as e:
            logger.error(f"Error in crawl_sync: {e}")
            return f"Error crawling webpage: {e}"

    def setup_tools(self) -> List[ToolFunctionDef]:
        """Setup and return the available tools for the AI model."""
        tool_crawl_link = ToolFunctionDef(
            name="Crawl a webpage",
            description="Crawls a webpage and extracts markdown content. You need to give an exact URL with https://.",
            parameters={"url": str},
            implementation=self.crawl_sync,
        )

        tool_search_pages = ToolFunctionDef(
            name="Wikipedia Search matching pages",
            description="Searches Wikipedia pages for a given query with the total number of pagenames to return set with limit. limit is capped at 500 and has a default value of 25.",
            parameters={"query": str, "limit": int},
            implementation=self.wikipedia_search,
        )

        tool_get_page_info = ToolFunctionDef(
            name="Wikipedia fetches a page",
            description="Gets information from a Wikipedia page for a given exact title. The title must be exact.",
            parameters={"page": str},
            implementation=self.get_wikipedia_page,
        )

        return [tool_search_pages, tool_get_page_info, tool_crawl_link]

    def wikipedia_search(self, query: str, limit: int = 25) -> List[str]:
        """
        Search Wikipedia API for pages matching a query.
        
        Args:
            query: Search query
            limit: Maximum number of results (capped at 500)
            
        Returns:
            List of page titles matching the query
        """
        try:
            url = 'https://en.wikipedia.org/w/api.php'
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': min(limit, 500),  # Cap at 500
            }
            logger.info(f"Searching Wikipedia for: {query}")
            
            # Cosmetic CLI message
            print(f"🔍 Searching Wikipedia for: {query}")
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('query', {}).get('search', [])
            result_list = [result['title'] for result in results]
            
            # Display context usage after tool call
            self.display_context_usage()
            
            return result_list
            
        except requests.RequestException as e:
            logger.error(f"Wikipedia search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in Wikipedia search: {e}")
            return []

    def get_wikipedia_page(self, page: str) -> str:
        """
        Get content from a Wikipedia page.
        
        Args:
            page: Exact title of the Wikipedia page
            
        Returns:
            Page content as plain text
        """
        try:
            url = 'https://en.wikipedia.org/w/api.php'
            params = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts',
                'explaintext': True,
                'titles': page
            }
            logger.info(f"Fetching Wikipedia page: {page}")
            
            # Cosmetic CLI message
            print(f"📖 Fetching Wikipedia page: {page}")
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            
            if not pages:
                result = "No page found."
            else:
                page_data = next(iter(pages.values()))
                result = page_data.get('extract', "No content found for the given page.")
            
            # Display context usage after tool call
            self.display_context_usage()
            
            return result
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Wikipedia page {page}: {e}")
            return f"Error fetching page: {e}"
        except Exception as e:
            logger.error(f"Unexpected error fetching Wikipedia page: {e}")
            return f"Unexpected error: {e}"

    def load_model(self) -> Optional[lms.LLM]:
        """
        Load or select an LLM model.
        
        Returns:
            Loaded LLM model instance
        """
        try:
            # Initialize lmstudio client if not already created
            if self.client is None:
                self.client = lms.Client()
            
            # Attempt to list already loaded models via lmstudio
            try:
                loaded_models = lms.list_loaded_models()
            except Exception:
                loaded_models = []
                
            if not loaded_models:
                logger.info("No models loaded")
                title = 'Please choose a model to load:'
                models = lms.list_downloaded_models("llm")
                
                if not models:
                    logger.error("No models available")
                    raise ValueError("No models found. Please download a model first.")
                
                choices = [m.model_key for m in models]
                # pick.pick returns (selected_option, index)
                if sys.stdin.isatty():
                    selected_tuple = pick.pick(choices, title)
                    selected_model = str(selected_tuple[0])
                else:
                    logger.warning("Non-interactive mode detected – selecting first model")
                    selected_model = choices[0]  # Extract and cast the selected model key
                # Load new model instance with the selected model_key
                model = self.client.llm.load_new_instance(
                    selected_model,
                    config={"contextLength": 32768}
                )
                logger.info("Model loaded successfully")
                return model
                
            # Use already loaded model via client
            model = self.client.llm.model()
            if model is None:
                logger.error("Failed to retrieve loaded model instance.")
                return None
                
            info = model.get_info()
            # Check if model is trained for tool use
            if not getattr(info, "trained_for_tool_use", False):
                logger.warning("Model not trained for tool use.")
                if input("Unload model? (y/N): ").lower() == 'y':
                    model.unload()
                    return self.load_model()
            else:
                logger.info(f"{info.identifier} is already loaded.")
                if input("Unload model? (y/N): ").lower() == 'y':
                    model.unload()
                    return self.load_model()
            return model
                    
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def calculate_context_usage(self, model: lms.LLM, chat: lms.Chat) -> Dict[str, Any]:
        """
        Calculate how much of the context window is being used.
        
        Args:
            model: The LLM model
            chat: The chat conversation
            
        Returns:
            Dictionary with context usage information
        """
        try:
            # Check if model and client are still active
            if not model or not self.client:
                logger.warning("Model or client not available for context calculation")
                return {"percentage": 0.0, "tokens_used": 0, "total_tokens": 0, "indicator": "❓"}
                
            formatted = model.apply_prompt_template(chat)
            tokens_used = len(model.tokenize(formatted))
            total_tokens = model.get_context_length()
            percentage = (tokens_used / total_tokens * 100) if total_tokens > 0 else 0.0
            
            # Create visual indicator based on usage percentage
            if percentage < 25:
                indicator = "🟢"  # Green - plenty of space
            elif percentage < 50:
                indicator = "🟡"  # Yellow - moderate usage
            elif percentage < 75:
                indicator = "🟠"  # Orange - getting full
            elif percentage < 90:
                indicator = "🔴"  # Red - almost full
            else:
                indicator = "⚠️"   # Warning - critical
            
            return {
                "percentage": percentage,
                "tokens_used": tokens_used,
                "total_tokens": total_tokens,
                "indicator": indicator
            }
        except Exception as e:
            logger.error(f"Error calculating context usage: {e}")
            return {"percentage": 0.0, "tokens_used": 0, "total_tokens": 0, "indicator": "❓"}

    def display_context_usage(self) -> None:
        """Display current context usage with visual indicator."""
        if not self.model or not self.chat:
            return
            
        usage_info = self.calculate_context_usage(self.model, self.chat)
        print(f"\n{usage_info['indicator']} Context: {usage_info['percentage']:.1f}% ({usage_info['tokens_used']}/{usage_info['total_tokens']} tokens)")

    def get_context_prefix(self) -> str:
        """Get context usage prefix for bot responses."""
        if not self.model or not self.chat:
            return "❓ Context: Unknown% | "
            
        usage_info = self.calculate_context_usage(self.model, self.chat)
        return f"{usage_info['indicator']} Context: {usage_info['percentage']:.1f}% | "

    def cleanup(self):
        """Clean up resources, like closing the LM Studio client."""
        if self.client:
            try:
                logger.info("Closing LM Studio client...")
                self.client.close()
                logger.info("LM Studio client closed.")
            except Exception as e:
                logger.error(f"Error closing LM Studio client: {e}")
        self.client = None

    def run(self) -> None:
        """Main execution loop for the AI Research Assistant."""
        try:
            # Load model and initialize chat
            self.model = self.load_model()
            if not self.model:
                logger.error("Failed to load model. Exiting.")
                return
                
            self.chat = lms.Chat("You are a task focused AI researcher")
            tools = self.setup_tools()
            
            logger.info("AI Research Assistant started. Type your queries or press Enter to exit.")
            
            while True:
                try:
                    user_input = input("\nYou (leave blank to exit): ")
                except EOFError:
                    print()
                    break
                    
                if not user_input.strip():
                    break
                    
                self.chat.add_user_message(user_input)
                
                # Show context usage before bot response
                context_prefix = self.get_context_prefix()
                print(f"{context_prefix}Bot: ", end="", flush=True)
                
                # Generate response
                self.model.act(
                    self.chat,
                    tools,
                    on_message=self.chat.append,
                    on_prediction_fragment=self.print_fragment,
                )
                print()  # New line after response
                
                # Show context usage after response (including any tool calls)
                self.display_context_usage()
                
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
            raise
        finally:
            logger.info("Cleaning up...")
            self.cleanup()


def main() -> None:
    """Entry point for the application."""
    assistant = AIResearchAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
