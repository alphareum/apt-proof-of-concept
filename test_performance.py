#!/usr/bin/env python3
"""
Test script to measure BodyCompositionAnalyzer initialization performance
"""

import sys
import time
from pathlib import Path

# Add src directory to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

def test_analyzer_initialization():
    """Test the initialization time of BodyCompositionAnalyzer"""
    print("Testing BodyCompositionAnalyzer initialization performance...")
    
    try:
        from apt_fitness.analyzers.body_composition import BodyCompositionAnalyzer
        
        print("Starting initialization...")
        start_time = time.time()
        
        # Initialize the analyzer
        analyzer = BodyCompositionAnalyzer()
        
        init_time = time.time() - start_time
        print(f"✅ Initialization completed in {init_time:.2f} seconds")
        
        # Test lazy loading of advanced models
        print("\nTesting lazy loading of advanced models...")
        start_time = time.time()
        
        analyzer._ensure_models_initialized()
        
        models_time = time.time() - start_time
        print(f"✅ Advanced models loaded in {models_time:.2f} seconds")
        
        print(f"\nTotal time: {init_time + models_time:.2f} seconds")
        print(f"Initial loading time (what user experiences): {init_time:.2f} seconds")
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_basic_functionality(analyzer):
    """Test basic functionality to ensure lazy loading works"""
    if not analyzer:
        return
        
    print("\nTesting basic functionality...")
    
    try:
        # Test that the analyzer has basic functionality
        print(f"- Pose model available: {analyzer.pose is not None}")
        print(f"- Segmentation model available: {analyzer.segmentation is not None}")
        print(f"- Database available: {analyzer.db is not None}")
        
        # Test if lazy initialization flags work
        print(f"- Models initialized: {analyzer._models_initialized}")
        print(f"- Preprocessing initialized: {analyzer._preprocessing_initialized}")
        
        print("✅ Basic functionality test passed")
        
    except Exception as e:
        print(f"❌ Error during functionality test: {e}")

if __name__ == "__main__":
    analyzer = test_analyzer_initialization()
    test_basic_functionality(analyzer)
