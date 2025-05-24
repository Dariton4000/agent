@echo off
echo Starting AI Research Agent...

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run setup.bat first to create the virtual environment.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate

REM Run the application
python main.py

REM Deactivate virtual environment
deactivate

echo.
echo AI Research Agent has been closed.
pause
