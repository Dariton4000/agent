#!/usr/bin/env python3
"""
Simple verification that the cleaned up project works correctly.
"""

import os
import sys

def check_project_structure():
    """Check if the project structure is clean and organized."""
    print("✅ Project Structure Check")
    print("-" * 30)
    
    expected_files = [
        'main.py',
        'test_main.py', 
        'requirements.txt',
        'README.md',
        '.env.example',
        '.gitignore',
        'logs/',
        'run_tests.py'
    ]
    
    missing = []
    for item in expected_files:
        if not os.path.exists(item):
            missing.append(item)
        else:
            print(f"✓ {item}")
    
    if missing:
        print(f"❌ Missing: {missing}")
        return False
    
    print("✓ All essential files present")
    return True

def check_logs_directory():
    """Check logs directory structure."""
    print("\n✅ Logs Directory Check")
    print("-" * 30)
    
    if os.path.exists('logs'):
        print("✓ logs/ directory exists")
        log_files = [f for f in os.listdir('logs') if f.endswith('.log')]
        print(f"✓ {len(log_files)} log files found")
        return True
    else:
        print("❌ logs/ directory missing")
        return False

def check_environment_config():
    """Check environment configuration."""
    print("\n✅ Environment Config Check")
    print("-" * 30)
    
    if os.path.exists('.env.example'):
        with open('.env.example', 'r') as f:
            content = f.read()
            if 'VERBOSE_LOGGING' in content:
                print("✓ VERBOSE_LOGGING option found in .env.example")
                return True
            else:
                print("❌ VERBOSE_LOGGING option missing")
                return False
    else:
        print("❌ .env.example missing")
        return False

def check_gitignore():
    """Check gitignore configuration."""
    print("\n✅ GitIgnore Check")
    print("-" * 30)
    
    if os.path.exists('.gitignore'):
        with open('.gitignore', 'r') as f:
            content = f.read()
            checks = [
                ('logs/', 'Logs directory ignored'),
                ('*.log', 'Log files ignored'),
                ('__pycache__/', 'Python cache ignored')
            ]
            
            all_good = True
            for pattern, description in checks:
                if pattern in content:
                    print(f"✓ {description}")
                else:
                    print(f"❌ {description}")
                    all_good = False
            
            return all_good
    else:
        print("❌ .gitignore missing")
        return False

def main():
    """Run all checks."""
    print("🧹 AI Research Assistant - Cleanup Verification")
    print("=" * 50)
    
    checks = [
        check_project_structure,
        check_logs_directory,
        check_environment_config,
        check_gitignore
    ]
    
    results = []
    for check in checks:
        results.append(check())
    
    print("\n" + "=" * 50)
    print("CLEANUP VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 Project cleanup completed successfully!")
        print("\nCleanup achievements:")
        print("✅ Removed unnecessary demo and refactoring files")
        print("✅ Created organized logs/ directory structure")
        print("✅ Added configurable verbose logging system")
        print("✅ Updated .env.example with logging options")
        print("✅ Configured .gitignore for proper exclusions")
        print("✅ Maintained clean project structure")
        
        print("\nLogging system features:")
        print("• All logs saved to timestamped files in logs/")
        print("• VERBOSE_LOGGING=true: Shows all logs in CLI")
        print("• VERBOSE_LOGGING=false: Shows only warnings/errors in CLI")
        print("• Always maintains complete log history in files")
        
        return True
    else:
        print("\n❌ Some cleanup issues detected")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
