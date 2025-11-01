#!/usr/bin/env python3
"""Simple test script to verify KroxAI-Mini package installation and structure."""

import sys

def test_package_import():
    """Test that the package can be imported."""
    print("Testing package import...")
    try:
        import kroxai_mini
        print("✓ kroxai_mini package imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import kroxai_mini: {e}")
        return False

def test_package_metadata():
    """Test package metadata and attributes."""
    print("\nTesting package metadata...")
    try:
        import kroxai_mini
        
        # Check __all__ attribute
        if hasattr(kroxai_mini, '__all__'):
            print(f"✓ Package exports: {kroxai_mini.__all__}")
        else:
            print("✗ Package missing __all__ attribute")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error checking metadata: {e}")
        return False

def test_lightweight_imports():
    """Test that lightweight components can be imported without heavy dependencies."""
    print("\nTesting lightweight imports (without numpy)...")
    
    # These should be importable if the package structure is correct
    # (though they may fail at runtime without numpy)
    lightweight_modules = [
        'kroxai_mini.tokenizer',
        'kroxai_mini.configs',
        'kroxai_mini.agent_protocol',
    ]
    
    all_passed = True
    for module in lightweight_modules:
        try:
            __import__(module)
            print(f"✓ {module} structure is valid")
        except ImportError as e:
            if 'numpy' in str(e).lower() or 'torch' in str(e).lower():
                print(f"⚠ {module} requires optional dependencies: {e}")
            else:
                print(f"✗ {module} import failed: {e}")
                all_passed = False
        except Exception as e:
            print(f"⚠ {module} import succeeded but initialization failed: {e}")
    
    return all_passed

def test_package_structure():
    """Test that the package structure is correct."""
    print("\nTesting package structure...")
    
    try:
        # Check that the package is installed
        import pkg_resources
        try:
            version = pkg_resources.get_distribution('kroxai-mini').version
            print(f"✓ Package kroxai-mini version {version} is installed")
        except pkg_resources.DistributionNotFound:
            print("✗ Package kroxai-mini is not installed")
            return False
        
    except ImportError:
        print("⚠ pkg_resources not available, skipping version check")
    
    # Check that the package directory exists
    try:
        import kroxai_mini
        import os
        package_path = os.path.dirname(kroxai_mini.__file__)
        print(f"✓ Package location: {package_path}")
        
        # List some expected modules
        expected_files = [
            '__init__.py',
            'tokenizer.py',
            'transformer.py',
            'data.py',
            'configs.py',
        ]
        
        for file in expected_files:
            file_path = os.path.join(package_path, file)
            if os.path.exists(file_path):
                print(f"  ✓ {file} exists")
            else:
                print(f"  ✗ {file} not found")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error checking package structure: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("KroxAI-Mini Installation Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Basic import
    results.append(("Package Import", test_package_import()))
    
    # Only continue if basic import works
    if results[0][1]:
        results.append(("Package Metadata", test_package_metadata()))
        results.append(("Package Structure", test_package_structure()))
        results.append(("Lightweight Imports", test_lightweight_imports()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Package structure is valid.")
        print("\nNote: Some imports may require optional dependencies like numpy or torch.")
        print("Install them with: pip install numpy torch")
        return 0
    else:
        print("✗ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
