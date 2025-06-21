#!/usr/bin/env python3
"""
Merge Multiple Dataset Reports
Combines CSV reports and JSONL training files from multiple build_and_report.py runs
"""

import os
import pandas as pd
import json
import glob
from datetime import datetime

def find_report_files(export_dir="data/exports"):
    """Find all CSV and JSONL files in exports directory."""
    csv_files = glob.glob(os.path.join(export_dir, "dataset_report_*.csv"))
    train_files = glob.glob(os.path.join(export_dir, "training_data_*.jsonl"))
    val_files = glob.glob(os.path.join(export_dir, "validation_data_*.jsonl"))
    
    return sorted(csv_files), sorted(train_files), sorted(val_files)

def merge_csv_reports(csv_files, output_path):
    """Merge multiple CSV reports into one consolidated report."""
    print(f"📊 Merging {len(csv_files)} CSV reports...")
    
    all_dataframes = []
    for csv_file in csv_files:
        print(f"   • Loading {os.path.basename(csv_file)}")
        df = pd.read_csv(csv_file)
        all_dataframes.append(df)
    
    # Concatenate all dataframes
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Remove duplicates based on chapter number (keep latest)
    merged_df = merged_df.drop_duplicates(subset=['Chapter'], keep='last')
    
    # Sort by chapter number
    merged_df = merged_df.sort_values('Chapter').reset_index(drop=True)
    
    # Save merged report
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✅ Merged CSV saved: {output_path}")
    print(f"📊 Total chapters: {len(merged_df)}")
    print(f"📈 Chapter range: {merged_df['Chapter'].min()} - {merged_df['Chapter'].max()}")
    
    return merged_df

def merge_jsonl_files(jsonl_files, output_path):
    """Merge multiple JSONL files into one."""
    print(f"📚 Merging {len(jsonl_files)} JSONL files...")
    
    all_examples = []
    seen_chapters = set()
    
    for jsonl_file in jsonl_files:
        print(f"   • Loading {os.path.basename(jsonl_file)}")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                
                # Extract chapter info to avoid duplicates
                user_content = example['messages'][1]['content']
                chapter_marker = user_content.split('\\n\\n')[0] if '\\n\\n' in user_content else user_content[:100]
                
                if chapter_marker not in seen_chapters:
                    all_examples.append(example)
                    seen_chapters.add(chapter_marker)
    
    # Save merged JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\\n')
    
    print(f"✅ Merged JSONL saved: {output_path}")
    print(f"📚 Total examples: {len(all_examples)}")
    
    return len(all_examples)

def main():
    """Main merge function."""
    print("🔗 Dataset Report Merger")
    print("=" * 50)
    
    # Find all report files
    csv_files, train_files, val_files = find_report_files()
    
    if not csv_files:
        print("❌ No CSV reports found in data/exports/")
        print("💡 Run build_and_report.py first to generate reports")
        return
    
    print(f"📂 Found files:")
    print(f"   • {len(csv_files)} CSV reports")
    print(f"   • {len(train_files)} training JSONL files")
    print(f"   • {len(val_files)} validation JSONL files")
    
    # Generate output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_csv = f"data/exports/merged_dataset_report_{timestamp}.csv"
    merged_train = f"data/exports/merged_training_data_{timestamp}.jsonl"
    merged_val = f"data/exports/merged_validation_data_{timestamp}.jsonl"
    
    # Merge CSV reports
    if csv_files:
        merged_df = merge_csv_reports(csv_files, merged_csv)
        
        # Show summary statistics
        print(f"\\n📊 **Merged Dataset Summary:**")
        print(f"   • Average BERT similarity: {merged_df['BERT_Similarity'].mean():.3f}")
        print(f"   • High quality chapters (similarity > 0.7): {len(merged_df[merged_df['BERT_Similarity'] > 0.7])}")
        print(f"   • Average English/Raw ratio: {merged_df['Eng_Raw_Ratio'].mean():.2f}")
    
    # Merge training files
    if train_files:
        train_count = merge_jsonl_files(train_files, merged_train)
    
    # Merge validation files
    if val_files:
        val_count = merge_jsonl_files(val_files, merged_val)
    
    print(f"\\n🎉 **Merge Complete!**")
    print(f"📁 Output files:")
    if csv_files:
        print(f"   • {os.path.basename(merged_csv)}")
    if train_files:
        print(f"   • {os.path.basename(merged_train)} ({train_count} examples)")
    if val_files:
        print(f"   • {os.path.basename(merged_val)} ({val_count} examples)")
    
    print(f"\\n💡 **Next Steps:**")
    print(f"   1. Review the merged CSV for data quality")
    print(f"   2. Use merged JSONL files for model training")
    print(f"   3. Consider removing individual report files to save space")

def show_current_files():
    """Show what files currently exist."""
    csv_files, train_files, val_files = find_report_files()
    
    print("📂 **Current Export Files:**")
    print("=" * 40)
    
    if csv_files:
        print("📊 CSV Reports:")
        for f in csv_files:
            print(f"   • {os.path.basename(f)}")
    
    if train_files:
        print("\\n🤖 Training Files:")
        for f in train_files:
            print(f"   • {os.path.basename(f)}")
    
    if val_files:
        print("\\n🔍 Validation Files:")
        for f in val_files:
            print(f"   • {os.path.basename(f)}")
    
    if not (csv_files or train_files or val_files):
        print("❌ No report files found")
        print("💡 Run: python build_and_report.py [args]")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        show_current_files()
    else:
        main()