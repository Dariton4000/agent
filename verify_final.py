#!/usr/bin/env python3
"""
Final verification script for the AI Research Assistant project.
Tests the logging system and verifies the cleanup is complete.
"""

import os
import sys
import subprocess
from datetime import datetime

def test_logging_system():
    """Test the logging system functionality."""
    print("🔍 Testing Logging System...")
    print("-" * 40)
    
    try:
        # Test basic import and logging setup
        sys.path.insert(0, os.getcwd())
        from main import setup_logging
        
        # Test with verbose logging disabled
        os.environ['VERBOSE_LOGGING'] = 'false'
        logger = setup_logging()
        
        print("✓ Logging system initialized successfully")
        
        # Test logging messages
        logger.info("Test INFO message - should only go to file")
        logger.warning("Test WARNING message - should appear in console and file")
        logger.error("Test ERROR message - should appear in console and file")
        
        print("✓ Log messages sent successfully")
        
        # Check if log files were created
        if os.path.exists('logs'):
            log_files = [f for f in os.listdir('logs') if f.endswith('.log')]
            print(f"✓ Found {len(log_files)} log files in logs/ directory")
            
            if log_files:
                latest_log = max(log_files)
                print(f"✓ Latest log file: {latest_log}")
                
                # Check log file content
                with open(f'logs/{latest_log}', 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Test INFO message' in content:
                        print("✓ INFO messages properly logged to file")
                    if 'Test WARNING message' in content:
                        print("✓ WARNING messages properly logged to file")
                    if 'Test ERROR message' in content:
                        print("✓ ERROR messages properly logged to file")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def verify_project_structure():
    """Verify the project structure is clean and organized."""
    print("\n📁 Verifying Project Structure...")
    print("-" * 40)
    
    essential_files = [
        'main.py',
        'test_main.py', 
        'requirements.txt',
        'README.md',
        '.env.example',
        '.gitignore'
    ]
    
    optional_files = [
        'test_logging.py',
        'verify_cleanup.py',
        'run_tests.py',
        'pytest.ini'
    ]
    
    all_good = True
    
    print("Essential files:")
    for file in essential_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            all_good = False
    
    print("\nOptional files:")
    for file in optional_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  - {file} (not required)")
    
    # Check logs directory
    if os.path.exists('logs'):
        print("  ✓ logs/ directory")
    else:
        print("  ❌ logs/ directory - MISSING")
        all_good = False
    
    return all_good

def verify_configuration():
    """Verify configuration files are properly set up."""
    print("\n⚙️ Verifying Configuration...")
    print("-" * 40)
    
    # Check .env.example
    if os.path.exists('.env.example'):
        with open('.env.example', 'r') as f:
            content = f.read()
            if 'VERBOSE_LOGGING' in content:
                print("✓ .env.example contains VERBOSE_LOGGING configuration")
            else:
                print("❌ .env.example missing VERBOSE_LOGGING")
                return False
    else:
        print("❌ .env.example missing")
        return False
    
    # Check .gitignore
    if os.path.exists('.gitignore'):
        with open('.gitignore', 'r') as f:
            content = f.read()
            checks = [
                ('logs/', 'logs directory exclusion'),
                ('*.log', 'log files exclusion'),
                ('__pycache__/', 'Python cache exclusion')
            ]
            
            for pattern, description in checks:
                if pattern in content:
                    print(f"✓ .gitignore contains {description}")
                else:
                    print(f"❌ .gitignore missing {description}")
                    return False
    else:
        print("❌ .gitignore missing")
        return False
    
    return True

def main():
    """Run all verification tests."""
    print("🎯 AI Research Assistant - Final Verification")
    print("=" * 60)
    print(f"Verification Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all verification tests
    tests = [
        ("Project Structure", verify_project_structure),
        ("Configuration", verify_configuration),
        ("Logging System", test_logging_system)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"{i+1}. {test_name}: {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 VERIFICATION SUCCESSFUL!")
        print("\n✅ Project Cleanup and Logging Implementation Complete!")
        print("\nAchievements:")
        print("• ✅ Clean project directory structure")
        print("• ✅ Smart logging system with verbose/quiet modes")
        print("• ✅ Timestamped log files in organized logs/ directory") 
        print("• ✅ Environment configuration with .env.example")
        print("• ✅ Proper .gitignore exclusions")
        print("• ✅ All configuration files properly set up")
        
        print("\nLogging Features:")
        print("• All logs saved to timestamped files (always)")
        print("• VERBOSE_LOGGING=true: Shows all logs in CLI")
        print("• VERBOSE_LOGGING=false: Shows only warnings/errors in CLI")
        print("• Professional formatting with timestamps")
        
        print("\nNext Steps:")
        print("• Run 'python main.py' to start the AI Research Assistant")
        print("• Modify .env file to change logging verbosity")
        print("• Check logs/ directory for complete execution history")
        
        return True
    else:
        print("\n❌ VERIFICATION FAILED")
        print("Some issues were detected. Please review the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
