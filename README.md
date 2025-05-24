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
- **Session Management**: Robust multi-session support with isolated memory, goals, and chat history per session.

## Features

- 🔍 Search Wikipedia for relevant pages with configurable result limits
- 📖 Fetch detailed content from Wikipedia pages with error handling
- 🌐 Crawl websites and extract content in markdown format
- 💬 Interactive conversation with AI models
- 📊 Real-time context usage monitoring
- 🔧 Robust error handling and recovery
- 📝 Smart logging system (verbose CLI mode can be enabled/disabled)
- 📁 Organized timestamped log files in `/logs` directory
- 💾 Session Management:
  - Create, load, and switch between multiple chat sessions.
  - Each session has its own isolated memory, goals, and chat history.
  - Sessions are persisted across application runs.
  - Auto-save of conversations.

## Requirements

- Python 3.8+ (compatible with LM-Studio and Crawl4AI)
- LM-Studio for AI model management
- Internet connection for Wikipedia and web crawling

## Installation

### Prerequisites

1. **Python 3.8+**: Ensure you have Python 3.8 or higher installed
2. **LM-Studio**: Install from [https://lmstudio.ai/](https://lmstudio.ai/)

### Setup Instructions

1. **Clone or download this repository**

2. **Create a virtual environment (Recommended):**

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install project dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Crawl4AI browser dependencies:**

   ```bash
   crawl4ai-setup
   ```

5. **Verify installation:**

   ```bash
   crawl4ai-doctor
   ```

6. **Configure environment variables (Optional):**

   Copy the environment example file:

   ```bash
   # Windows
   copy .env.example .env

   # macOS/Linux
   cp .env.example .env
   ```

   Edit `.env` with your preferred settings:

   ```bash
   # Enable verbose logging in CLI (shows all logs in terminal)
   VERBOSE_LOGGING="true"

   # Disable verbose logging (only errors/warnings in CLI, all logs in files)
   VERBOSE_LOGGING="false"
   ```

**Logging Behavior:**

- **All logs are always saved** to timestamped files in the `logs/` directory
- **Verbose mode (`VERBOSE_LOGGING="true"`)**: Shows all `INFO`, `WARNING`, `ERROR`, and `CRITICAL` logs in the terminal.
- **Quiet mode (`VERBOSE_LOGGING="false"`)**: Shows only `WARNING`, `ERROR`, and `CRITICAL` logs in the terminal.

## Usage

1. **Start LM-Studio**:
   - Open LM-Studio.
   - Load your desired AI model.
   - Start the server (usually on `http://localhost:1234`).

2. **Run the AI Research Assistant:**

   ```bash
   python main.py
   ```

   Or, if using a virtual environment:

   ```bash
   # Windows
   venv\Scripts\python.exe main.py

   # macOS/Linux
   venv/bin/python main.py
   ```

3. **Interact with the AI**:
   - The application will guide you through session selection or creation.
   - Type your research queries or commands.
   - Use "exit" or "quit" to end the session.

## Available Tools

The AI assistant has access to the following tools:

- **Wikipedia Search**: Search for Wikipedia pages.
  - `Wikipedia Search matching pages`: Finds page titles.
  - `Wikipedia fetches a page`: Gets content from an exact page title.
- **Web Crawling**: Crawl websites.
  - `Crawl a webpage`: Extracts markdown content from a URL.
- **Memory Management**:
  - `Create memory`: Store important information.
  - `Recall memories`: Retrieve stored information.
- **Goal Management**:
  - `Create goal`: Set research objectives.
  - `Get active goals`: View current objectives.

## System Prompt Overview

The AI is guided by a system prompt that outlines its capabilities, tool usage best practices, memory strategies, and a general research workflow. This prompt encourages the AI to:

- Use multiple tools in a single response if needed.
- Store important findings in its memory.
- Create and track goals for complex tasks.
- Leverage its memory and goals to provide context-aware responses.

## Project Structure

```text
.
├── .env.example         # Example environment variables
├── .gitignore           # Git ignore file
├── LICENSE              # Project license
├── README.md            # This file
├── main.py              # Main application script
├── requirements.txt     # Python dependencies
├── pytest.ini           # Pytest configuration
├── assets/              # Static assets (e.g., images for README)
│   └── image.png
├── logs/                # Directory for log files (auto-created)
│   └── agent_YYYYMMDD_HHMMSS.log
├── sessions/            # Directory for session data (auto-created)
│   ├── index.json       # Index of all sessions
│   └── [session_id]/    # Directory for a specific session
│       ├── chat.json    # Chat history for the session
│       ├── goals.json   # Goals for the session
│       ├── index.json   # Memory/Goal ID tracking for the session
│       └── memories.json # Memories for the session
├── tests/               # Test scripts (if you add them)
│   ├── test_assistant.py
│   └── test_tools.py
└── venv/                # Python virtual environment (if created)
```

## Testing

(If you have tests, describe how to run them here. Example:)

```bash
pytest
```

Or to run with verbose output:

```bash
pytest -v
```

## Troubleshooting

### Common Issues

1. **Python not found after activation**:
   - Ensure your virtual environment is correctly activated.
   - Verify `python` or `python3` points to the venv interpreter (`which python` or `where python`).

2. **Import errors**:
   - Make sure all dependencies in `requirements.txt` are installed in your active virtual environment (`pip install -r requirements.txt`).
   - If you added new libraries, update `requirements.txt` (`pip freeze > requirements.txt`).

3. **LM-Studio Connection Issues**:
   - Confirm LM-Studio is running and the server is active.
   - Check the server address and port in your LM-Studio settings (usually `http://localhost:1234`).
   - Ensure no firewall is blocking the connection.

4. **Crawl4AI Issues**:
   - Run `crawl4ai-doctor` to diagnose browser setup problems.
   - Ensure `crawl4ai-setup` completed successfully.

### Logging

- **Log files**: Detailed logs are stored in the `logs/` directory, named `agent_<timestamp>.log`. These logs contain all `INFO` level messages and above, regardless of the `VERBOSE_LOGGING` setting.
- **Console output**:
  - If `VERBOSE_LOGGING="true"`, the console will show `INFO` messages and above.
  - If `VERBOSE_LOGGING="false"`, the console will only show `WARNING`, `ERROR`, and `CRITICAL` messages. This helps keep the terminal output clean while still capturing detailed information in the log files.

### Debugging Tips

- **Enable Verbose Logging**: Set `VERBOSE_LOGGING="true"` in your `.env` file to get more detailed output in the console.
- **Check Log Files**: Always refer to the timestamped log files in the `logs/` directory for complete error messages and context.
- **Examine Tool Calls**: The logs (and verbose console output) will show when the AI attempts to use tools and the results of those calls. This is crucial for debugging tool-related issues.
- **Monitor Context Window**: The `Context: X.X% (Y/Z tokens)` message shows how much of the AI's context window is being used. If it gets too high, the AI might lose older parts of the conversation.
- **Session Data**: Inspect the files in the `sessions/[session_id]/` directory (`chat.json`, `memories.json`, `goals.json`) to understand the state of a particular session.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bugs, features, or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
