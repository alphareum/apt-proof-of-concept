"""
Cleanup Script for AI Fitness Assistant

This script helps remove unnecessary files and keep only what's needed for the fitness app.
"""

import os
import shutil
import sys
from pathlib import Path

def main():
    """Main cleanup function."""
    print("🧹 AI Fitness Assistant - Cleanup Script")
    print("=" * 50)
    
    # Files to KEEP for the fitness app
    keep_files = {
        # Core fitness app files
        'fitness_app.py',
        'requirements_fitness.txt',
        'run_fitness_app.py',
        'setup_fitness_app.bat',
        'setup_fitness_app.ps1',
        'FITNESS_README.md',
        'QUICK_START.md',
        
        # Git and environment files (optional to keep)
        '.git',
        '.gitignore',
        '.env',
        '.env.template',
        
        # Virtual environments (optional to keep)
        'fitness_venv',
        
        # This cleanup script itself
        'cleanup_fitness.py'
    }
    
    # Files to REMOVE (APT/original project files)
    remove_files = {
        # Original APT application files
        'ai_processor.py',
        'app.py',
        'config.py',
        'demo.py',
        'frontend_improved.py',
        'main.py',
        'models.py',
        
        # APT documentation
        'CHANGELOG.md',
        'DEPLOYMENT.md',
        'README.md',  # Keep FITNESS_README.md instead
        
        # APT configuration files
        'requirements.txt',  # Keep requirements_fitness.txt instead
        'pyproject.toml',
        'setup.py',
        'uv.lock',
        
        # Docker files (for APT deployment)
        'docker-compose.yml',
        'Dockerfile',
        'Dockerfile.frontend',
        '.dockerignore',
        
        # APT directories
        'instance',
        'logs',
        'nginx',
        'pose_data',
        'uploads',
        '__pycache__',
        'venv',  # Keep fitness_venv instead
        '.venv',
        
        # Python version file
        '.python-version'
    }
    
    # Get current directory
    current_dir = Path.cwd()
    
    print(f"📂 Working in: {current_dir}")
    print()
    
    # Show what will be removed
    print("🗑️  Files/folders to be REMOVED:")
    files_to_remove = []
    for item in remove_files:
        item_path = current_dir / item
        if item_path.exists():
            files_to_remove.append(item_path)
            file_type = "📁" if item_path.is_dir() else "📄"
            print(f"   {file_type} {item}")
    
    print()
    print("✅ Files/folders to be KEPT:")
    for item in keep_files:
        item_path = current_dir / item
        if item_path.exists():
            file_type = "📁" if item_path.is_dir() else "📄"
            print(f"   {file_type} {item}")
    
    print()
    
    if not files_to_remove:
        print("✨ Nothing to remove - directory is already clean!")
        return
    
    # Ask for confirmation
    print(f"⚠️  About to remove {len(files_to_remove)} items.")
    print("   This action cannot be undone!")
    print()
    
    while True:
        response = input("🤔 Do you want to proceed? (yes/no/list): ").lower().strip()
        
        if response in ['yes', 'y']:
            break
        elif response in ['no', 'n']:
            print("❌ Cleanup cancelled.")
            return
        elif response in ['list', 'l']:
            print("\n📋 Detailed list of items to remove:")
            for item_path in files_to_remove:
                size = get_size_info(item_path)
                print(f"   • {item_path.name} {size}")
            print()
        else:
            print("   Please enter 'yes', 'no', or 'list'")
    
    # Perform cleanup
    print("\n🧹 Starting cleanup...")
    
    removed_count = 0
    for item_path in files_to_remove:
        try:
            if item_path.is_dir():
                shutil.rmtree(item_path)
                print(f"   📁 Removed directory: {item_path.name}")
            else:
                item_path.unlink()
                print(f"   📄 Removed file: {item_path.name}")
            removed_count += 1
        except Exception as e:
            print(f"   ❌ Failed to remove {item_path.name}: {e}")
    
    print(f"\n✅ Cleanup completed! Removed {removed_count} items.")
    print("\n🏋️‍♀️ Your fitness app is now ready!")
    print("\n🚀 To run the app:")
    print("   python run_fitness_app.py")
    print("   or")
    print("   setup_fitness_app.bat")

def get_size_info(path):
    """Get size information for a file or directory."""
    try:
        if path.is_file():
            size = path.stat().st_size
            if size < 1024:
                return f"({size} bytes)"
            elif size < 1024 * 1024:
                return f"({size / 1024:.1f} KB)"
            else:
                return f"({size / (1024 * 1024):.1f} MB)"
        elif path.is_dir():
            # Count files in directory
            try:
                file_count = sum(1 for _ in path.rglob('*') if _.is_file())
                return f"({file_count} files)"
            except:
                return "(directory)"
    except:
        return ""
    return ""

if __name__ == "__main__":
    main()
