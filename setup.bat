@echo off
echo Setting up AI Research Agent...

echo.
echo 1. Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo 2. Activating virtual environment...
call venv\Scripts\activate

echo.
echo 3. Upgrading pip...
python -m pip install --upgrade pip

echo.
echo 4. Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo 5. Setting up Crawl4AI...
crawl4ai-setup

echo.
echo 6. Running health check...
crawl4ai-doctor

echo.
echo 7. Copying environment file...
copy .env.example .env

echo.
echo 8. Validating setup...
python validate_setup.py

echo.
echo Setup complete!
echo.
echo To run the application:
echo 1. Activate the virtual environment: venv\Scripts\activate
echo 2. Run the agent: python main.py
echo.
echo Or simply use: run.bat
echo.
pause
