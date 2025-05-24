# AI Research Agent

A professional AI research assistant with access to Wikipedia and web crawling capabilities. This tool allows you to search Wikipedia, fetch page content, and crawl websites to gather information for research purposes.

## ✨ Recent Improvements

The codebase has been significantly improved for better maintainability and reliability:

- **Object-Oriented Design**: Restructured into a clean `AIResearchAssistant` class
- **Comprehensive Error Handling**: Robust error handling throughout all functions
- **Configurable Logging**: Smart logging system with verbose/quiet modes and timestamped log files
- **Extensive Documentation**: Full docstrings for all methods and classes
- **Improved Wikipedia Search**: Fixed to return all results instead of just the first one
- **Configuration Management**: Environment variable support with `.env` files
- **Comprehensive Testing**: Full test suite with unit, integration, and error handling tests
- **Better Tool Organization**: Tools are now properly organized and configured
- **Context Management**: Improved context window usage calculation and display
- **Clean Project Structure**: Organized logs directory and cleaned up unnecessary files

## Features

- 🔍 Search Wikipedia for relevant pages with configurable result limits
- 📖 Fetch detailed content from Wikipedia pages with error handling
- 🌐 Crawl websites and extract content in markdown format
- 💬 Interactive conversation with AI models
- 📊 Real-time context usage monitoring
- 🔧 Robust error handling and recovery
- 📝 Smart logging system (verbose CLI mode can be enabled/disabled)
- 📁 Organized timestamped log files in `/logs` directory

## Requirements

- Python 3.8+ (compatible with LM-Studio and Crawl4AI)
- LM-Studio for AI model management
- Internet connection for Wikipedia and web crawling

## Installation

### Prerequisites

1. Install LM-Studio from [https://lmstudio.ai/](https://lmstudio.ai/)

2. Install project dependencies:

```bash
pip install -r requirements.txt
```

3. Install Crawl4AI dependencies:

```bash
pip install crawl4ai[all]
crawl4ai-setup
```

4. Verify installation:

```bash
crawl4ai-doctor
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Modify `.env` with your preferred settings (optional):

```bash
# Enable verbose logging in CLI (shows all logs in terminal)
VERBOSE_LOGGING="true"

# Disable verbose logging (only errors/warnings in CLI, all logs in files)
VERBOSE_LOGGING="false"
```

**Logging Behavior:**
- **All logs are always saved** to timestamped files in the `logs/` directory
- **VERBOSE_LOGGING="true"**: Shows all logs in CLI (good for development/debugging)
- **VERBOSE_LOGGING="false"**: Shows only warnings/errors in CLI (clean production mode)

## Usage

### Quick Start

1. Start LM-Studio and enable the Server:

   ![LM Studio UI tutorial](assets/image.png)

   Or via command line:
   ```bash
   lms server start
   ```

2. Make sure you have a model downloaded in LM-Studio (qwen3-8b recommended)

3. Run the agent:
   ```bash
   python main.py
   ```

4. Start asking research questions!

### Testing

Run the comprehensive test suite:

```bash
python run_tests.py
```

Or run basic validation:

```bash
python validate_refactoring.py
```

## Architecture

The refactored codebase follows modern Python best practices:

```
main.py                 # Main AIResearchAssistant class
├── AIResearchAssistant # Main class with all functionality
├── Web Crawling        # Async web crawling with error handling  
├── Wikipedia API       # Search and content retrieval
├── Tool Management     # LM-Studio tool configuration
└── Context Management  # Token usage and conversation handling

test_main.py           # Comprehensive test suite
├── Unit Tests         # Individual function testing
├── Integration Tests  # End-to-end workflow testing
└── Error Handling     # Edge case and error testing
```

## How It Works

The agent integrates several technologies:

- **LM Studio API**: Communicates with local AI models for natural language processing
- **Wikipedia API**: Searches and retrieves information from Wikipedia with proper error handling
- **Crawl4AI**: Extracts clean markdown content from web pages asynchronously
- **AsyncIO**: Handles concurrent operations for better performance
- **Logging**: Provides detailed execution logs for debugging and monitoring

## API Reference

### Main Class: AIResearchAssistant

#### Core Methods

- `crawl_webpage(url: str) -> str`: Asynchronously crawl a webpage and return markdown
- `wikipedia_search(query: str, limit: int = 25) -> List[str]`: Search Wikipedia pages
- `get_wikipedia_page(page: str) -> str`: Fetch content from a specific Wikipedia page
- `load_model() -> lms.LLM`: Load or select an LLM model with user interaction
- `run() -> None`: Main execution loop with interactive chat interface

#### Tool Integration

The assistant automatically configures tools for the AI model:
- Wikipedia search tool with configurable result limits
- Wikipedia page content retrieval tool  
- Web page crawling tool with markdown extraction

## Development

### Running Tests

The project includes a comprehensive test suite:

```bash
# Run all tests with detailed output
python run_tests.py

# Run basic validation (no external dependencies)
python validate_refactoring.py

# Run specific test categories (requires pytest)
pytest test_main.py::TestAIResearchAssistant -v  # Unit tests
pytest test_main.py::TestIntegration -v          # Integration tests
pytest test_main.py::TestErrorHandling -v        # Error handling
```

### Code Quality

The refactored code includes:
- Type hints for better IDE support and documentation
- Comprehensive error handling with specific exception types
- Structured logging with configurable levels
- Docstrings following Google/NumPy style
- Separation of concerns with clear class structure

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed via `pip install -r requirements.txt`
2. **Model Loading Issues**: Ensure LM-Studio is running and has models downloaded
3. **Network Errors**: Check internet connection for Wikipedia and web crawling
4. **Context Window**: Monitor the context usage ratio displayed during chat

### Logging

Check `agent.log` for detailed execution logs including:
- Model loading status
- API request/response information  
- Error details and stack traces
- Performance metrics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

### Planned Features

- 🧠 **Memory and Goal System**: Implement persistent memory to overcome context window limitations
- 🔍 **Enhanced Search**: Add support for additional search engines beyond Wikipedia  
- 📁 **File Operations**: Add ability to save and load research sessions
- 🔄 **Conversation History**: Implement conversation history management
- 🎨 **Web Interface**: Develop a web-based UI for easier interaction
- 📊 **Analytics**: Add research session analytics and insights

### Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
