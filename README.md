# Way of the Devil Translation Dataset

## Project Overview
Creating aligned Chinese-English parallel text dataset for "Way of the Devil" web novel for ML training purposes.

## 📁 Essential Files & Directories

### Core Scripts
- **`robust_scraper.py`** - Main scraper with metadata support and smart resuming
- **`process_epub.py`** - Extracts English chapters from EPUB files  
- **`review_ui_with_ai.py`** - Streamlit UI for manual review with 3 panes (Chinese/English/AI translation)
- **`quick_resume.py`** - Quick analysis tool to find missing chapters

### Data Files
- **`CLAUDE.md`** - Project instructions and current status
- **`requirements.txt`** - Python dependencies
- **`scraping_metadata.json`** - Auto-generated metadata with URL mappings (enables smart resuming)
- **`alignment_map.json`** - Chapter alignment mapping for UI
- **`Way_of_the_Devil_1-772.epub`** - Original English translation source

### Data Directories
- **`english_chapters/`** - 772 extracted English chapters (English-Chapter-0001.txt to English-Chapter-0772.txt)
- **`novel_content_dxmwx_complete/`** - Chinese raw chapters (Chapter-0640 to Chapter-1213 range)

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Resume scraping (if needed):**
   ```bash
   python robust_scraper.py
   ```

3. **Check for gaps:**
   ```bash
   python quick_resume.py
   ```

4. **Launch review UI:**
   ```bash
   streamlit run review_ui_with_ai.py
   ```

## 🔧 How It Works

1. **`robust_scraper.py`** automatically builds `scraping_metadata.json` as it works
2. **Smart resuming** - checks metadata first, skips known chapters instantly
3. **Crash-resistant** - saves metadata every 10 chapters
4. **Gap detection** - `quick_resume.py` finds missing chapters without network requests
5. **Manual review** - UI shows Chinese/English/AI translation side-by-side

## 📊 Current Status

- ✅ **English chapters**: 772 extracted and ready
- ✅ **Chinese chapters**: ~570 chapters scraped  
- ✅ **Smart scraper**: Metadata-driven with instant resuming
- ✅ **Review UI**: 3-pane interface with Gemini AI translation

## 💡 Key Features

- **Metadata-driven**: Never loses progress, can resume from any chapter
- **Smart skipping**: Instant detection of existing files via metadata
- **Robust parsing**: Handles malformed Chinese chapter titles
- **Quality control**: Manual review UI with AI translation for verification