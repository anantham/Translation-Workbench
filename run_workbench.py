#!/usr/bin/env python3
"""
Translation Framework Workbench Launcher
Easy launcher for the multi-page Streamlit application
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import pandas
        import plotly
        print("✅ Core dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("📥 Install with: pip install -r requirements.txt")
        return False

def launch_workbench():
    """Launch the Translation Framework Workbench."""
    print("🚀 Translation Framework Workbench")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Verify we're in the right directory
    if not os.path.exists("pages/1_📖_Data_Review_Alignment.py"):
        print("❌ Data Review page not found. Please run from the project directory.")
        return
    
    if not os.path.exists("utils.py"):
        print("❌ utils.py not found. Please ensure all files are present.")
        return
    
    if not os.path.exists("pages"):
        print("❌ pages/ directory not found. Please ensure all files are present.")
        return
    
    print("🎯 Starting multi-page workbench...")
    print("📊 Available pages:")
    print("  • 📖 Data Review & Alignment")
    print("  • 🤖 Fine-tuning Workbench")
    print("  • 🧪 Pluralistic Translation Lab")
    print("  • 📈 Experimentation Analysis")
    print()
    print("🌐 The workbench will open in your default browser")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "pages/1_📖_Data_Review_Alignment.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Workbench stopped by user")
    except Exception as e:
        print(f"❌ Error launching workbench: {e}")

if __name__ == "__main__":
    launch_workbench()