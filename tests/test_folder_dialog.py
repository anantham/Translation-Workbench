"""
Test script to verify folder selection dialog functionality.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog

def test_folder_dialog():
    """Test the folder selection dialog."""
    print("Testing folder selection dialog...")
    
    try:
        # Create root window and hide it
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Test Chinese folder selection
        print("Opening Chinese folder selection dialog...")
        initial_dir = os.path.abspath("../data/novels") if os.path.exists("../data/novels") else os.path.expanduser("~")
        
        folder_path = filedialog.askdirectory(
            title="Select Chinese Chapters Directory",
            initialdir=initial_dir
        )
        
        if folder_path:
            print(f"Selected folder: {folder_path}")
            
            # Check if it contains txt files
            if os.path.exists(folder_path):
                txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
                print(f"Found {len(txt_files)} .txt files in selected folder")
                
                # Show first few files as examples
                if txt_files:
                    print("Sample files:")
                    for i, filename in enumerate(txt_files[:5]):
                        print(f"  {i+1}. {filename}")
                    if len(txt_files) > 5:
                        print(f"  ... and {len(txt_files) - 5} more files")
        else:
            print("No folder selected")
        
        root.destroy()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_folder_dialog()