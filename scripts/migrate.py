#!/usr/bin/env python3
"""
Migration script to move old files to legacy folder
"""

import shutil
from pathlib import Path
import os


def main():
    """Main migration function."""
    print("🔄 APT Fitness Assistant - Migration Script")
    print("=" * 50)
    
    # Create legacy directory
    legacy_dir = Path("legacy")
    legacy_dir.mkdir(exist_ok=True)
    
    # Files to move to legacy
    legacy_files = [
        "app.py",
        "fitness_app.py", 
        "fitness_app_enhanced.py",
        "body_composition_analyzer.py",
        "body_composition_api.py",
        "body_composition_demo.py",
        "body_composition_ui.py",
        "database.py",
        "enhanced_exercise_database.py",
        "enhanced_progress_tracking.py",
        "enhanced_recommendation_system.py",
        "enhanced_ui.py",
        "models.py",
        "nutrition_planner.py",
        "progress_analytics.py",
        "recipe_database.py",
        "recipe_manager.py",
        "recommendation_engine.py",
        "social_features.py",
        "ui_components.py",
        "utils.py",
        "workout_planner.py",
        "workout_planner_ui.py",
        "run_fitness_app.py",
        "test_app.py",
        "test_enhanced_app.py",
        "config.py"
    ]
    
    # Move files
    moved_count = 0
    for file_name in legacy_files:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                shutil.move(str(file_path), str(legacy_dir / file_name))
                print(f"✅ Moved {file_name} to legacy/")
                moved_count += 1
            except Exception as e:
                print(f"❌ Failed to move {file_name}: {e}")
    
    # Move batch/script files
    script_files = [
        "cleanup_fitness.bat",
        "cleanup_fitness.py", 
        "run_enhanced_app.bat",
        "run-app.bat",
        "run-app.ps1",
        "setup_body_composition.bat",
        "setup_body_composition.sh",
        "setup_fitness_app.bat",
        "setup_fitness_app.ps1",
        "setup-uv.bat",
        "setup-uv.ps1",
        "start_app.bat"
    ]
    
    for file_name in script_files:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                shutil.move(str(file_path), str(legacy_dir / file_name))
                print(f"✅ Moved {file_name} to legacy/")
                moved_count += 1
            except Exception as e:
                print(f"❌ Failed to move {file_name}: {e}")
    
    # Move documentation files
    doc_files = [
        "BODY_COMPOSITION_README.md",
        "ENHANCED_README.md", 
        "ENHANCED_WORKOUT_PLANNER_README.md",
        "FITNESS_README.md",
        "IMPROVEMENTS_SUMMARY.md",
        "QUICK_START.md",
        "README_UV.md"
    ]
    
    for file_name in doc_files:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                shutil.move(str(file_path), str(legacy_dir / file_name))
                print(f"✅ Moved {file_name} to legacy/")
                moved_count += 1
            except Exception as e:
                print(f"❌ Failed to move {file_name}: {e}")
    
    # Move requirements files
    req_files = [
        "requirements_enhanced.txt",
        "requirements_fitness.txt", 
        "requirements_windows.txt"
    ]
    
    for file_name in req_files:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                shutil.move(str(file_path), str(legacy_dir / file_name))
                print(f"✅ Moved {file_name} to legacy/")
                moved_count += 1
            except Exception as e:
                print(f"❌ Failed to move {file_name}: {e}")
    
    print(f"\n🎉 Migration completed! Moved {moved_count} files to legacy/")
    print("\n📂 Project Structure:")
    print("   src/apt_fitness/    - New modular codebase")
    print("   main.py            - New application entry point")
    print("   legacy/            - Old files (preserved)")
    print("\n🚀 Ready to use the new structure!")


if __name__ == "__main__":
    main()
