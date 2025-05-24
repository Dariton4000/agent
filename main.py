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
from typing import Optional, List, Dict, Any, Tuple
import json
from pathlib import Path
import uuid

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
        
        # Chat session management
        self.current_session_id: Optional[str] = None
        self.session_title: Optional[str] = None
        
        # Memory management (now per-session)
        self.memory_dir = Path("memory")
        self.sessions_dir = Path("sessions")
        self.memory_dir.mkdir(exist_ok=True)
        self.sessions_dir.mkdir(exist_ok=True)
        
        # Global files (for session index)
        self.sessions_index_file = self.sessions_dir / "index.json"
        
        # Session-specific files (will be set when session is loaded/created)
        self.memories_file: Optional[Path] = None
        self.goals_file: Optional[Path] = None
        self.chat_file: Optional[Path] = None
        self.memory_index_file: Optional[Path] = None
        
        # Initialize session management
        self._initialize_session_files()
        
        logger.info("AIResearchAssistant initialized with chat session management")
    
    def _initialize_session_files(self) -> None:
        """Initialize session management files."""
        if not self.sessions_index_file.exists():
            self.sessions_index_file.write_text(json.dumps({
                "sessions": {},
                "last_session_id": None
            }))

    def _initialize_memory_files(self) -> None:
        """Initialize memory files for current session."""
        if not self.memories_file or not self.goals_file or not self.memory_index_file:
            return
            
        if not self.memories_file.exists():
            self.memories_file.write_text(json.dumps([]))
        if not self.goals_file.exists():
            self.goals_file.write_text(json.dumps([]))
        if not self.memory_index_file.exists():
            self.memory_index_file.write_text(json.dumps({"next_id": 1}))

    def _set_session_files(self, session_id: str) -> None:
        """Set file paths for a specific session."""
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        self.memories_file = session_dir / "memories.json"
        self.goals_file = session_dir / "goals.json"
        self.chat_file = session_dir / "chat.json"
        self.memory_index_file = session_dir / "index.json"

    def create_new_session(self, title: Optional[str] = None) -> str:
        """
        Create a new chat session.
        
        Args:
            title: Optional title for the session
            
        Returns:
            Session ID of the new session
        """
        session_id = str(uuid.uuid4())[:8]  # Short UUID
        timestamp = datetime.now().isoformat()
        
        if not title:
            title = f"Chat {datetime.now().strftime('%m/%d %H:%M')}"
        
        # Set up session files
        self._set_session_files(session_id)
        self._initialize_memory_files()
        
        # Update sessions index
        try:
            sessions_index = json.loads(self.sessions_index_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            sessions_index = {"sessions": {}, "last_session_id": None}
        
        sessions_index["sessions"][session_id] = {
            "title": title,
            "created_at": timestamp,
            "last_accessed": timestamp,
            "message_count": 0
        }
        sessions_index["last_session_id"] = session_id
        
        self.sessions_index_file.write_text(json.dumps(sessions_index, indent=2))
        
        # Set current session
        self.current_session_id = session_id
        self.session_title = title
        
        # Initialize empty chat history
        self._save_chat_history([])
        
        print(f"📝 Created new session: {title} (ID: {session_id})")
        logger.info(f"Created new session: {session_id} - {title}")
        
        return session_id

    def load_session(self, session_id: str) -> bool:
        """
        Load an existing chat session.
        
        Args:
            session_id: ID of the session to load
            
        Returns:
            True if session loaded successfully, False otherwise
        """
        try:
            # Check if session exists
            sessions_index = json.loads(self.sessions_index_file.read_text())
            if session_id not in sessions_index.get("sessions", {}):
                print(f"❌ Session {session_id} not found")
                return False
              # Set up session files
            self._set_session_files(session_id)
            
            # Initialize session files if they don't exist
            self._initialize_memory_files()
              # Ensure chat file exists with empty array if it doesn't
            if not self.chat_file or not self.chat_file.exists():
                self._save_chat_history([])
                print(f"📄 Initialized empty chat history for session {session_id}")
            
            # Load session info
            session_info = sessions_index["sessions"][session_id]
            self.current_session_id = session_id
            self.session_title = session_info["title"]
            
            # Update last accessed time
            session_info["last_accessed"] = datetime.now().isoformat()
            sessions_index["last_session_id"] = session_id
            self.sessions_index_file.write_text(json.dumps(sessions_index, indent=2))
            
            print(f"📂 Loaded session: {self.session_title} (ID: {session_id})")
            logger.info(f"Loaded session: {session_id} - {self.session_title}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            print(f"❌ Error loading session: {e}")
            return False

    def switch_to_session(self, session_id: str) -> str:
        """
        Switch to an existing chat session.
        
        Args:
            session_id: ID of the session to switch to
            
        Returns:
            Status message about the session switch
        """
        try:
            if self.load_session(session_id):
                # Load chat history for display
                messages = self._load_chat_history()
                message_count = len(messages)
                
                print(f"🔄 Switched to session: {self.session_title}")
                logger.info(f"Switched to session: {session_id}")
                
                # Display recent chat history (last 3 messages)
                if messages:
                    print("\n📜 Recent chat history:")
                    recent_messages = messages[-6:]  # Last 3 exchanges (user + bot)
                    for msg in recent_messages:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        timestamp = msg.get("timestamp", "")[:16].replace("T", " ")
                        
                        if role == "user":
                            print(f"  [{timestamp}] You: {content}")
                        elif role == "assistant":
                            print(f"  [{timestamp}] Bot: {content}")
                
                return f"Successfully switched to session '{self.session_title}' (ID: {session_id}). Found {message_count} messages."
            else:
                return f"Failed to switch to session {session_id}. Session not found or corrupted."
                
        except Exception as e:
            logger.error(f"Failed to switch to session {session_id}: {e}")
            return f"Error switching to session: {e}"

    def list_sessions(self) -> str:
        """
        List all available chat sessions.
        
        Returns:
            Formatted string of available sessions
        """
        try:
            sessions_index = json.loads(self.sessions_index_file.read_text())
            sessions = sessions_index.get("sessions", {})
            
            if not sessions:
                return "No chat sessions found."
            
            # Sort by last accessed (most recent first)
            sorted_sessions = sorted(
                sessions.items(),
                key=lambda x: x[1].get("last_accessed", ""),
                reverse=True
            )
            
            result_parts = [f"Available Chat Sessions ({len(sessions)}):"]
            for session_id, info in sorted_sessions:
                last_accessed = info.get("last_accessed", "Unknown")[:10]  # Date only
                message_count = info.get("message_count", 0)
                current_marker = " 🔹 (Current)" if session_id == self.current_session_id else ""
                
                result_parts.append(f"\n📝 {session_id}: {info['title']}{current_marker}")
                result_parts.append(f"   Last accessed: {last_accessed}")
                result_parts.append(f"   Messages: {message_count}")
            
            return "\n".join(result_parts)
            
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return f"Error listing sessions: {e}"

    def _load_chat_history(self) -> List[Dict[str, Any]]:
        """Load chat history for current session."""
        if not self.chat_file or not self.chat_file.exists():
            return []
        
        try:
            return json.loads(self.chat_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_chat_history(self, messages: List[Dict[str, Any]]) -> None:
        """Save chat history for current session."""
        if not self.chat_file:
            return
        
        self.chat_file.write_text(json.dumps(messages, indent=2))
        
        # Update message count in session index
        try:
            sessions_index = json.loads(self.sessions_index_file.read_text())
            if self.current_session_id in sessions_index.get("sessions", {}):
                sessions_index["sessions"][self.current_session_id]["message_count"] = len(messages)
                sessions_index["sessions"][self.current_session_id]["last_accessed"] = datetime.now().isoformat()
                self.sessions_index_file.write_text(json.dumps(sessions_index, indent=2))
        except Exception as e:
            logger.error(f"Failed to update session index: {e}")

    def _auto_save_message(self, role: str, content: str) -> None:
        """Auto-save a message to chat history."""
        if not self.current_session_id:
            return
        
        messages = self._load_chat_history()
        messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._save_chat_history(messages)

    def _ensure_session_active(self) -> bool:
        """Ensure there's an active session for memory operations."""
        if not self.current_session_id:
            # Create a default session if none exists
            self.create_new_session("Default Session")
        return self.current_session_id is not None

    def _load_memories(self) -> List[Dict[str, Any]]:
        """Load memories from file."""
        if not self._ensure_session_active() or not self.memories_file:
            return []
        try:
            return json.loads(self.memories_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_memories(self, memories: List[Dict[str, Any]]) -> None:
        """Save memories to file."""
        if not self._ensure_session_active() or not self.memories_file:
            return
        self.memories_file.write_text(json.dumps(memories, indent=2))

    def _load_goals(self) -> List[Dict[str, Any]]:
        """Load goals from file."""
        if not self._ensure_session_active() or not self.goals_file:
            return []
        try:
            return json.loads(self.goals_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_goals(self, goals: List[Dict[str, Any]]) -> None:
        """Save goals to file."""
        if not self._ensure_session_active() or not self.goals_file:
            return
        self.goals_file.write_text(json.dumps(goals, indent=2))

    def _get_next_id(self) -> int:
        """Get next available ID for memories/goals."""
        if not self._ensure_session_active() or not self.memory_index_file:
            return 1
        try:
            index = json.loads(self.memory_index_file.read_text())
            next_id = index.get("next_id", 1)
            index["next_id"] = next_id + 1
            self.memory_index_file.write_text(json.dumps(index))
            return next_id
        except (json.JSONDecodeError, FileNotFoundError):
            self.memory_index_file.write_text(json.dumps({"next_id": 2}))
            return 1

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
                  # Handle the crawl result - should be a CrawlResultContainer
                if hasattr(result, 'success') and result.success:
                    # Try to get the best available content format
                    if hasattr(result, 'markdown') and result.markdown:
                        return result.markdown
                    elif hasattr(result, 'cleaned_html') and result.cleaned_html:
                        return result.cleaned_html
                    elif hasattr(result, 'text') and result.text:
                        return result.text
                    else:
                        return "Content extracted but no readable text found."
                else:
                    error_msg = getattr(result, 'error_message', 'Unknown error occurred')
                    return f"Failed to crawl webpage: {error_msg}"
        except Exception as e:
            logger.error(f"Failed to crawl {url}: {e}")
            raise

    def crawl_sync(self, url: str) -> str:
        """Synchronous wrapper for crawl_webpage."""
        try:
            # Handle both running and non-running event loop scenarios
            try:
                # Check if we're already in an event loop
                asyncio.get_running_loop()
                # If we are, we need to use a different approach
                # For now, return an error message suggesting async usage
                return "Error: Cannot run sync crawl from within async context. Use crawl_webpage() directly."
            except RuntimeError:
                # No running loop → safe to create one
                try:
                loop = asyncio.get_running_loop()
                result = loop.run_until_complete(self.crawl_webpage(url))
            except RuntimeError:
                # No running loop → safe to create one
                result = asyncio.run(self.crawl_webpage(url))
                # Note: Context will be displayed after the full response is complete
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
        )        # Memory management tools
        tool_create_memory = ToolFunctionDef(
            name="Create memory",
            description="Create a new memory entry to remember important information. Use this to store key findings, insights, or important facts for later recall.",
            parameters={"title": str, "content": str, "tags": str},
            implementation=self.create_memory,
        )

        tool_create_goal = ToolFunctionDef(
            name="Create goal",
            description="Create a new goal or objective. Use this to set tasks, research objectives, or long-term aims. Priority can be: low, medium, high, critical.",
            parameters={"title": str, "description": str, "priority": str},
            implementation=self.create_goal,
        )

        tool_recall_memories = ToolFunctionDef(
            name="Recall memories",
            description="Search and retrieve stored memories based on query or tags. Use this to find previously stored information.",
            parameters={"query": str, "tags": str, "limit": int},
            implementation=self.recall_memories,
        )

        tool_get_goals = ToolFunctionDef(
            name="Get active goals",
            description="Retrieve all currently active goals and objectives.",
            parameters={},
            implementation=self.get_active_goals,
        )

        return [tool_search_pages, tool_get_page_info, tool_crawl_link, tool_create_memory, tool_create_goal, tool_recall_memories, tool_get_goals]

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
            
            # Note: Context will be displayed after the full response is complete
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
            
            # Note: Context will be displayed after the full response is complete
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

    def create_memory(self, title: str, content: str, tags: str = "") -> str:
        """
        Create a new memory entry.
        
        Args:
            title: Brief title for the memory
            content: Detailed content of the memory
            tags: Comma-separated tags for categorization
            
        Returns:
            Confirmation message with memory ID
        """
        try:
            memory_id = self._get_next_id()
            timestamp = datetime.now().isoformat()
            
            new_memory = {
                "id": memory_id,
                "title": title,
                "content": content,
                "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
                "created_at": timestamp,
                "accessed_count": 0,
                "last_accessed": timestamp
            }
            
            memories = self._load_memories()
            memories.append(new_memory)
            self._save_memories(memories)
            
            # Enhanced logging with content preview and tags
            content_preview = content[:100] + "..." if len(content) > 100 else content
            tags_display = f" | Tags: {tags}" if tags else ""
            print(f"💾 Storing memory #{memory_id}: '{title}' - {content_preview}{tags_display}")
            logger.info(f"Created memory #{memory_id}: {title}")
            
            return f"Memory created successfully with ID #{memory_id}: '{title}'"
            
        except Exception as e:
            logger.error(f"Failed to create memory: {e}")
            return f"Error creating memory: {e}"

    def create_goal(self, title: str, description: str, priority: str = "medium") -> str:
        """
        Create a new goal.
        
        Args:
            title: Brief title for the goal
            description: Detailed description of the goal
            priority: Priority level (low, medium, high, critical)
            
        Returns:
            Confirmation message with goal ID
        """
        try:
            goal_id = self._get_next_id()
            timestamp = datetime.now().isoformat()
            
            # Validate priority
            valid_priorities = ["low", "medium", "high", "critical"]
            if priority.lower() not in valid_priorities:
                priority = "medium"
            
            new_goal = {
                "id": goal_id,
                "title": title,
                "description": description,
                "priority": priority.lower(),
                "status": "active",
                "created_at": timestamp,
                "progress_notes": []
            }
            
            goals = self._load_goals()
            goals.append(new_goal)
            self._save_goals(goals)
            
            # Enhanced logging with priority indicator and description preview
            priority_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(priority.lower(), "🟡")
            desc_preview = description[:80] + "..." if len(description) > 80 else description
            print(f"🎯 Setting goal #{goal_id}: '{title}' {priority_emoji} {priority.upper()} - {desc_preview}")
            logger.info(f"Created goal #{goal_id}: {title}")
            
            return f"Goal created successfully with ID #{goal_id}: '{title}' (Priority: {priority})"
            
        except Exception as e:
            logger.error(f"Failed to create goal: {e}")
            return f"Error creating goal: {e}"

    def recall_memories(self, query: str = "", tags: str = "", limit: int = 5) -> str:
        """
        Recall memories based on query or tags.
        
        Args:
            query: Search query for title/content (optional)
            tags: Comma-separated tags to filter by (optional)
            limit: Maximum number of memories to return
            
        Returns:
            Formatted string of matching memories
        """
        try:
            memories = self._load_memories()
            
            if not memories:
                return "No memories found."
            
            # Filter by tags if specified
            if tags:
                tag_list = [tag.strip().lower() for tag in tags.split(",") if tag.strip()]
                memories = [m for m in memories if any(tag in [t.lower() for t in m.get("tags", [])] for tag in tag_list)]
            
            # Filter by query if specified
            if query:
                query_lower = query.lower()
                memories = [m for m in memories if 
                           query_lower in m.get("title", "").lower() or 
                           query_lower in m.get("content", "").lower()]
            
            # Sort by creation date (newest first)
            memories.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            # Limit results
            memories = memories[:limit]
            
            if not memories:
                return f"No memories found matching criteria."
            
            # Update access counts
            for memory in memories:
                memory["accessed_count"] = memory.get("accessed_count", 0) + 1
                memory["last_accessed"] = datetime.now().isoformat()
            
            self._save_memories(self._load_memories())  # Save updated access counts
            
            # Format results
            result_parts = [f"Found {len(memories)} memory(ies):"]
            for memory in memories:
                tags_str = ", ".join(memory.get("tags", [])) if memory.get("tags") else "No tags"
                result_parts.append(f"\n#{memory['id']}: {memory['title']}")
                result_parts.append(f"Tags: {tags_str}")
                result_parts.append(f"Content: {memory['content']}")
                result_parts.append(f"Created: {memory['created_at'][:10]}")
                result_parts.append("---")
            print(f"🧠 Retrieving {len(memories)} memory(ies) matching criteria: query='{query}', tags='{tags}'")
            return "\n".join(result_parts)
            
        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return f"Error recalling memories: {e}"

    def get_active_goals(self) -> str:
        """
        Get all active goals.
        
        Returns:
            Formatted string of active goals
        """
        try:
            goals = self._load_goals()
            active_goals = [g for g in goals if g.get("status", "active") == "active"]
            
            if not active_goals:
                return "No active goals found."
            
            # Sort by priority (critical > high > medium > low)
            priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            active_goals.sort(key=lambda x: priority_order.get(x.get("priority", "medium"), 2), reverse=True)
            
            # Format results
            result_parts = [f"Active Goals ({len(active_goals)}):"]
            for goal in active_goals:
                priority_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(goal.get("priority", "medium"), "🟡")
                result_parts.append(f"\n{priority_emoji} #{goal['id']}: {goal['title']}")
                result_parts.append(f"Description: {goal['description']}")
                result_parts.append(f"Priority: {goal.get('priority', 'medium').title()}")
                result_parts.append(f"Created: {goal['created_at'][:10]}")
                if goal.get("progress_notes"):
                    result_parts.append(f"Notes: {len(goal['progress_notes'])} progress updates")
                result_parts.append("---")
            print(f"🎯 Loading {len(active_goals)} active goal(s) by priority")
            return "\n".join(result_parts)
            
        except Exception as e:
            logger.error(f"Failed to get active goals: {e}")
            return f"Error getting active goals: {e}"

    def get_memory_context(self) -> str:
        """
        Get formatted memory and goals context for injection into model.
        
        Returns:
            Formatted context string with recent memories and active goals
        """
        try:
            # Get recent memories (last 3)
            memories = self._load_memories()
            recent_memories = sorted(memories, key=lambda x: x.get("created_at", ""), reverse=True)[:3]
            
            # Get active goals
            goals = self._load_goals()
            active_goals = [g for g in goals if g.get("status", "active") == "active"]
            
            context_parts = []
            
            if active_goals:
                context_parts.append("ACTIVE GOALS:")
                for goal in active_goals[:3]:  # Limit to top 3 goals
                    priority = goal.get("priority", "medium").upper()
                    context_parts.append(f"- [{priority}] {goal['title']}: {goal['description']}")
                context_parts.append("")
            
            if recent_memories:
                context_parts.append("RECENT MEMORIES:")
                for memory in recent_memories:
                    tags = ", ".join(memory.get("tags", [])) if memory.get("tags") else ""
                    context_parts.append(f"- {memory['title']}: {memory['content']}")
                    if tags:
                        context_parts.append(f"  Tags: {tags}")
                context_parts.append("")
            
            return "\n".join(context_parts) if context_parts else ""
            
        except Exception as e:
            logger.error(f"Failed to get memory context: {e}")
            return ""

    def _handle_session_selection(self) -> None:
        """Handle session selection at application startup."""
        try:
            # Check if there are existing sessions
            sessions_index = {}
            if self.sessions_index_file.exists():
                sessions_index = json.loads(self.sessions_index_file.read_text())
            
            sessions = sessions_index.get("sessions", {})
            last_session_id = sessions_index.get("last_session_id")
            
            if not sessions:
                # No sessions exist, create a default one
                print("🆕 No existing sessions found. Creating your first session...")
                session_title = input("Enter a title for your first session (or press Enter for default): ").strip()
                if not session_title:
                    session_title = "First Session"
                self.create_new_session(session_title)
                return
            
            # Show available sessions
            print("\n" + "="*50)
            print("🎯 AI Research Assistant - Session Manager")
            print("="*50)
            
            print(f"\nAvailable Sessions ({len(sessions)}):")
            sorted_sessions = sorted(
                sessions.items(),
                key=lambda x: x[1].get("last_accessed", ""),
                reverse=True
            )
            
            for i, (session_id, info) in enumerate(sorted_sessions, 1):
                last_accessed = info.get("last_accessed", "Unknown")[:10]
                message_count = info.get("message_count", 0)
                current_marker = " (Last used)" if session_id == last_session_id else ""
                print(f"{i}. {session_id}: {info['title']}{current_marker}")
                print(f"   Messages: {message_count}, Last accessed: {last_accessed}")
            
            print(f"\n{len(sessions) + 1}. Create new session")
            
            # Get user choice
            while True:
                try:
                    choice = input(f"\nSelect a session (1-{len(sessions) + 1}): ").strip()
                    
                    if not choice:
                        # Default to last used session if available
                        if last_session_id and last_session_id in sessions:
                            choice = "1"  # Last used is first in sorted list
                        else:
                            continue
                    
                    choice_num = int(choice)
                    
                    if 1 <= choice_num <= len(sessions):
                        # Load existing session
                        selected_session_id = sorted_sessions[choice_num - 1][0]
                        if self.load_session(selected_session_id):
                            break
                        else:
                            print("❌ Failed to load session. Please try another.")
                    elif choice_num == len(sessions) + 1:
                        # Create new session
                        session_title = input("Enter title for new session: ").strip()
                        if not session_title:
                            session_title = f"Session {datetime.now().strftime('%m/%d %H:%M')}"
                        self.create_new_session(session_title)
                        break
                    else:
                        print(f"❌ Please enter a number between 1 and {len(sessions) + 1}")
                        
                except ValueError:
                    print("❌ Please enter a valid number")
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    sys.exit(0)
            
        except Exception as e:
            logger.error(f"Error in session selection: {e}")
            print(f"❌ Error during session selection: {e}")
            print("🆕 Creating default session...")
            self.create_new_session("Default Session")

    def run(self) -> None:
        """Main execution loop for the AI Research Assistant."""
        try:
            # Load model and initialize chat
            self.model = self.load_model()
            if not self.model:
                logger.error("Failed to load model. Exiting.")
                return
            
            # Handle session selection at startup
            self._handle_session_selection()            # Create system message with memory context
            memory_context = self.get_memory_context()
            system_message = """You are an advanced AI Research Assistant with sophisticated memory management capabilities.

CORE CAPABILITIES:
- Research and information gathering through Wikipedia and web crawling
- Dynamic memory system for storing important findings, insights, and facts
- Goal management for tracking objectives and research tasks

TOOL USAGE GUIDELINES:
- You can use MULTIPLE tools in a single response when needed
- Chain tool calls together for comprehensive research (e.g., search → fetch → store memory)
- Store important findings immediately using memory tools
- Create goals to track research objectives and maintain focus

MEMORY BEST PRACTICES:
- Store key findings with descriptive titles and relevant tags
- Create memories for important facts, quotes, statistics, and insights
- Use goals to track research objectives and progress
- Tag memories appropriately for easy retrieval (e.g., "research", "statistics", "source")

RESEARCH WORKFLOW:
1. When asked about a topic, search for information first
2. Store important findings in memory with relevant tags
3. Create goals if the task involves multiple steps
4. Provide comprehensive responses based on gathered information"""
            
            if memory_context:
                system_message += f"\n\nCURRENT CONTEXT:\n{memory_context}"
                system_message += "\n\nConsider your existing memories and goals when responding. Use memory tools to build upon previous research."
            else:
                system_message += "\n\nStart building your knowledge base using memory and goal tools as you research."
            
            # Load existing chat history if available
            chat_history = self._load_chat_history()
            if chat_history:
                # Create chat with system message
                self.chat = lms.Chat(system_message)                # Add historical messages (but not system messages)
                for msg in chat_history:
                    if msg.get("role") == "user":
                        self.chat.add_user_message(msg.get("content", ""))
                    elif msg.get("role") == "assistant":
                        # For assistant messages, we'll add them as system context since we can't add model messages directly
                        pass  # Skip adding assistant messages to avoid confusion
                print(f"📜 Loaded {len(chat_history)} previous messages")
            else:
                self.chat = lms.Chat(system_message)
            
            tools = self.setup_tools()
            
            print(f"🧠 Memory system initialized for session: {self.session_title}")
            if memory_context:
                print("📚 Loaded existing memories and goals")
            logger.info(f"AI Research Assistant started with session {self.current_session_id}. Type your queries or press Enter to exit.")
            
            while True:
                try:
                    user_input = input("\nYou (leave blank to exit): ")
                except EOFError:
                    print()
                    break
                    
                if not user_input.strip():
                    break
                
                # Auto-save user message
                self._auto_save_message("user", user_input)
                    
                self.chat.add_user_message(user_input)
                
                # Show context usage before bot response
                context_prefix = self.get_context_prefix()
                print(f"{context_prefix}Bot: ", end="", flush=True)
                  # Capture bot response for auto-save
                bot_response_parts = []
                
                def capture_fragment(fragment, round_index: int = 0):
                    bot_response_parts.append(fragment.content)
                    self.print_fragment(fragment, round_index)
                
                # Generate response
                self.model.act(
                    self.chat,
                    tools,
                    on_message=self.chat.append,
                    on_prediction_fragment=capture_fragment,
                )
                print()  # New line after response
                
                # Auto-save bot response
                bot_response = "".join(bot_response_parts)
                if bot_response.strip():
                    self._auto_save_message("assistant", bot_response)
                
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
