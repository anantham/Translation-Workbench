#!/usr/bin/env python3
"""
Batch Processing Guide
Shows how build_and_report.py handles multiple runs and output files
"""

def explain_current_behavior():
    print("🎯 **Current Behavior: Separate Files per Run**")
    print("=" * 55)
    
    print("When you run:")
    print("```bash")
    print("python build_and_report.py 50      # First 50 chapters")
    print("python build_and_report.py 51 100  # Chapters 51-100")
    print("```")
    
    print("\\n📁 **You get SEPARATE files:**")
    print("```")
    print("data/exports/")
    print("├── dataset_report_20250620_140530.csv      # First run (50 chapters)")
    print("├── training_data_20250620_140530.jsonl")
    print("├── validation_data_20250620_140530.jsonl")
    print("├── dataset_report_20250620_141205.csv      # Second run (chapters 51-100)")
    print("├── training_data_20250620_141205.jsonl")
    print("└── validation_data_20250620_141205.jsonl")
    print("```")

def show_merge_options():
    print("\\n🔗 **Option 1: Manual Merge (Recommended)**")
    print("=" * 50)
    
    print("After multiple runs, merge them:")
    print("```bash")
    print("# See what files exist")
    print("python scripts/utils/merge_reports.py --list")
    print("")
    print("# Merge all reports into one")
    print("python scripts/utils/merge_reports.py")
    print("```")
    
    print("✅ **Benefits:**")
    print("   • Keep individual runs for debugging")
    print("   • Combine only when ready")
    print("   • Remove duplicates automatically")
    print("   • Full control over the process")

def show_automatic_option():
    print("\\n🤖 **Option 2: Automatic Merge Mode**")
    print("=" * 45)
    
    print("I can modify build_and_report.py to have a --merge flag:")
    print("```bash")
    print("python build_and_report.py 50           # Creates separate files")
    print("python build_and_report.py 51 100       # Creates separate files")
    print("python build_and_report.py --merge-all  # Combines everything")
    print("```")
    
    print("⚖️ **Trade-offs:**")
    print("   ✅ More convenient")
    print("   ❌ Less control over what gets merged")
    print("   ❌ Harder to debug individual runs")

def show_recommendations():
    print("\\n💡 **Recommended Workflow**")
    print("=" * 35)
    
    print("**Phase 1: Test with small batches**")
    print("```bash")
    print("python build_and_report.py 5        # Test first 5")
    print("python scripts/utils/merge_reports.py --list  # Check output")
    print("```")
    
    print("\\n**Phase 2: Process in manageable chunks**")
    print("```bash")
    print("python build_and_report.py 50       # Chapters 1-50")
    print("python build_and_report.py 51 100   # Chapters 51-100")
    print("python build_and_report.py 101 150  # Chapters 101-150")
    print("# ... continue as needed")
    print("```")
    
    print("\\n**Phase 3: Merge when satisfied**")
    print("```bash")
    print("python scripts/utils/merge_reports.py")
    print("```")
    
    print("\\n**Phase 4: Clean up (optional)**")
    print("```bash")
    print("# Keep only the merged files, remove individual ones")
    print("rm data/exports/dataset_report_2025*.csv")
    print("rm data/exports/training_data_2025*.jsonl")
    print("# (Keep the merged_* files)")
    print("```")

def show_benefits():
    print("\\n🎯 **Why This Approach Works Well**")
    print("=" * 40)
    
    print("✅ **Incremental Progress**: See results after each batch")
    print("✅ **Error Isolation**: If one batch fails, others are safe")
    print("✅ **Quality Control**: Review each batch before merging")
    print("✅ **Flexibility**: Rerun just the problematic chapters")
    print("✅ **Memory Efficient**: Process large datasets in chunks")
    print("✅ **Resume Capability**: Pick up where you left off")

if __name__ == "__main__":
    explain_current_behavior()
    show_merge_options()
    show_automatic_option()
    show_recommendations()
    show_benefits()
    
    print("\\n🚀 **Ready to Start?**")
    print("Try: python build_and_report.py 5")