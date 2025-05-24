#!/usr/bin/env python3
"""
Test runner script for the AI Research Assistant.
"""

import subprocess
import sys
import os


def run_tests():
    """Run the test suite."""
    print("Running AI Research Assistant Tests...")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("pytest not found. Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        import pytest
    
    # Run unit tests
    print("\n1. Running Unit Tests...")
    result_unit = pytest.main([
        "test_main.py::TestAIResearchAssistant",
        "-v",
        "--tb=short"
    ])
    
    # Run integration tests
    print("\n2. Running Integration Tests...")
    result_integration = pytest.main([
        "test_main.py::TestIntegration",
        "-v",
        "--tb=short"
    ])
    
    # Run error handling tests
    print("\n3. Running Error Handling Tests...")
    result_error = pytest.main([
        "test_main.py::TestErrorHandling",
        "-v",
        "--tb=short"
    ])
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Unit Tests: {'PASSED' if result_unit == 0 else 'FAILED'}")
    print(f"Integration Tests: {'PASSED' if result_integration == 0 else 'FAILED'}")
    print(f"Error Handling Tests: {'PASSED' if result_error == 0 else 'FAILED'}")
    
    overall_result = result_unit + result_integration + result_error
    print(f"\nOverall: {'PASSED' if overall_result == 0 else 'FAILED'}")
    
    return overall_result


def run_main_script():
    """Test running the main script."""
    print("\n" + "=" * 50)
    print("Testing Main Script Execution...")
    
    try:
        # Import the main module to check for syntax errors
        import main
        print("✓ Main script imports successfully")
        
        # Check if the main class can be instantiated
        assistant = main.AIResearchAssistant()
        print("✓ AIResearchAssistant class can be instantiated")
        
        # Check if tools can be set up
        tools = assistant.setup_tools()
        print(f"✓ Tools setup successful ({len(tools)} tools available)")
        
        print("✓ Main script validation PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Main script validation FAILED: {e}")
        return False


if __name__ == "__main__":
    print("AI Research Assistant - Test Suite")
    print("=" * 50)
    
    # Test main script
    main_ok = run_main_script()
    
    # Run tests
    test_result = run_tests()
    
    # Final result
    print("\n" + "=" * 50)
    if main_ok and test_result == 0:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the output above.")
        sys.exit(1)
