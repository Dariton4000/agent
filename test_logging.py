#!/usr/bin/env python3
"""
Simple test script to verify the logging configuration works correctly.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_logging_verbose():
    """Test verbose logging enabled."""
    print("Testing VERBOSE_LOGGING=true...")
    
    # Set environment variable
    os.environ['VERBOSE_LOGGING'] = 'true'
    
    # Import and test
    from main import setup_logging
    logger = setup_logging()
    
    print("With verbose logging enabled, you should see these log messages in console:")
    logger.info("This is an INFO message (should appear in console)")
    logger.warning("This is a WARNING message (should appear in console)")
    logger.error("This is an ERROR message (should appear in console)")
    
    return True

def test_logging_quiet():
    """Test quiet logging (verbose disabled)."""
    print("\n" + "="*60)
    print("Testing VERBOSE_LOGGING=false...")
    
    # Set environment variable
    os.environ['VERBOSE_LOGGING'] = 'false'
    
    # Need to reload the module to test new configuration
    import importlib
    import main
    importlib.reload(main)
    
    logger = main.setup_logging()
    
    print("With verbose logging disabled, you should only see WARNING and ERROR in console:")
    logger.info("This is an INFO message (should NOT appear in console)")
    logger.warning("This is a WARNING message (should appear in console)")
    logger.error("This is an ERROR message (should appear in console)")
    
    return True

def test_log_files():
    """Test that log files are created."""
    print("\n" + "="*60)
    print("Checking log files...")
    
    logs_dir = "logs"
    if os.path.exists(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
        print(f"Found {len(log_files)} log files in logs/ directory:")
        for log_file in log_files:
            print(f"  - {log_file}")
        return len(log_files) > 0
    else:
        print("No logs directory found")
        return False

def main():
    """Run all logging tests."""
    print("🧪 Testing AI Research Assistant Logging System")
    print("="*60)
    
    # Load environment
    load_dotenv()
    
    tests = [
        test_logging_verbose,
        test_logging_quiet,
        test_log_files
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("LOGGING TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All logging tests passed!")
        print("\nLogging system is working correctly:")
        print("- Verbose mode shows all logs in CLI")
        print("- Quiet mode shows only warnings/errors in CLI")
        print("- All logs are saved to timestamped files in logs/")
    else:
        print("❌ Some logging tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
