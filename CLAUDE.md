# Claude Code Configuration

## Project: Way of the Devil Translation System

### CURRENT STATUS (2025-06-20)
**MAJOR PROGRESS ACHIEVED:**
- ✅ Raw data scraping working perfectly (dxmwx.org)
- ✅ English EPUB processing complete (772 chapters extracted)
- ✅ Reconnaissance and framework validation complete
- ✅ Complete MLOps Translation Workbench implemented
- ✅ Multi-page Streamlit application with modular architecture
- ✅ Dataset alignment and curation tools perfected
- ✅ Fine-tuning workbench with Google AI SDK integration
- ✅ Experimentation lab with model comparison and evaluation
- 🔄 **CURRENT TASK**: Debug build_and_report.py and expand to Pluralistic Translation Workbench

### Project Overview
Goal: Create aligned Chinese-English parallel text dataset for "Way of the Devil" web novel
- **Chinese Source**: https://www.dxmwx.org/read/43713_33325507.html (1,213 total chapters)
- **English Source**: Way_of_the_Devil_1-772.epub (772 chapters extracted)
- **Target**: Align first 772 chapters for ML training dataset

### Environment Setup
```bash
cd "/Users/adityaprasad/Library/CloudStorage/OneDrive-IndianInstituteofScience/Documents/Ongoing/Project 1 Wuxia"
source venv/bin/activate
```

### Completed Components
1. **main_scraper.py** - Working static scraper for dxmwx.org
2. **process_epub.py** - Successfully extracted 772 English chapters
3. **recon_scraper.py** - Reconnaissance tool (validated both sites)

### Current Data Status
- **novel_content_dxmwx/**: 5 Chinese chapters (proof of concept)
- **english_chapters/**: 772 English chapters (English-Chapter-0001.txt to English-Chapter-0772.txt)

### IMMEDIATE NEXT STEPS
1. **Scale Chinese scraping**: Modify main_scraper.py to get 772+ chapters
2. **Chapter alignment**: Verify Chinese Ch.1 = English Ch.1, etc.
3. **Manual spot-checking**: Claude can read Chinese to verify alignment quality

Our first task is reconnaissance. The script below, recon_scraper.py, is designed to probe a target URL and gather as much information as possible. It does not perform full scraping.
File: recon_scraper.py
Purpose: Given a URL, this script performs two types of analysis:
Static Analysis: Makes a simple HTTP GET request to fetch the raw HTML, headers, and status code. This tells us what a basic bot sees.
Dynamic Analysis: Uses a headless browser (Selenium) to load the page, execute JavaScript, and then capture the rendered HTML and a screenshot. This tells us what a user's browser sees.
The output will be saved into a directory named after the website's domain, allowing us to compare results for different novels.


Analyze Output: After running, a new directory (e.g., output_www_example-novel-site_com) will contain:
static_analysis.html: Raw HTML from the server.
static_headers.json: Server response headers.
dynamic_analysis.html: HTML after JavaScript rendering.
dynamic_screenshot.png: A visual snapshot of the page as seen by the browser.
Report Back: Please provide me with a summary of the findings or the contents of these files so I can analyze the site's behavior and design the appropriate scraper.
6. Testing Instructions
We will use pytest for testing.
Tests will reside in a /tests directory.
We will use saved HTML files (from the recon script) as test fixtures to avoid hitting the network during tests. This ensures our tests are fast and deterministic.
7. Project-Specific Notes & Warnings
Ethical Scraping: Always introduce delays (time.sleep()) in any looping scraper to avoid overwhelming the server. We will use a respectful User-Agent.
Failure Modes: We are actively tracking failure modes. Key ones to look for are:
IP Blocks: The server starts refusing connections after several requests.
Cloudflare/JS Challenges: The static HTML contains "Please wait..." or "Checking your browser..." while the dynamic HTML shows the real content.
Paywalls/Login Walls: The content is explicitly hidden behind a login form. The screenshot will be very useful for identifying this.
API Character Limit: Remember the downstream goal is to feed data into an LLM with a ~40,000 character limit per example. We will need to keep this in mind during the final data processing stage.
Action Item for You: The Reconnaissance Script
Here is the code for recon_scraper.py. Please save it, set up your environment, and run it on the two initial URLs:
https://wtr-lab.com/en/serie-5467/extreme-demon?tab=toc
https://www.dxmwx.org/read/43713_33325507.html


## DEBUGGING METHODOLOGY - EVIDENCE-DRIVEN DIAGNOSIS

### The Scientific Debugging Process (2025-07-15)

**🚨 CRITICAL RULE: Never patch based on assumptions. Always gather evidence first.**

#### Step 1: Formulate Multiple Uncorrelated Hypotheses
When encountering a bug like `'NoneType' object is not callable`, generate 3-5 different hypotheses that could explain the error:

**Example for NovelCool scraper error:**
1. **Import/Module Loading Issue**: New code not loaded due to Python cache
2. **Method Call on None**: `content_div.copy()` returns None or method doesn't exist  
3. **Missing Method**: `_clean_content_container()` method not defined or accessible
4. **BeautifulSoup Version**: Method compatibility issues between BS4 versions
5. **Exception in Method**: Error inside `_clean_content_container()` not caught

#### Step 2: Add Verbose Debug Logging to Collect Evidence
Don't guess - instrument the code to gather data:

```python
def extract_content(self, soup):
    logger.debug("[NOVELCOOL] Starting content extraction")
    
    # HYPOTHESIS TESTING: Check what we actually found
    content_div = soup.find('p', class_='chapter-start-mark')
    logger.debug(f"[NOVELCOOL] content_div type: {type(content_div)}")
    logger.debug(f"[NOVELCOOL] content_div value: {content_div}")
    
    if content_div:
        logger.debug(f"[NOVELCOOL] content_div.copy method exists: {hasattr(content_div, 'copy')}")
        logger.debug(f"[NOVELCOOL] _clean_content_container method exists: {hasattr(self, '_clean_content_container')}")
        
        # Test each step individually
        try:
            copied_div = content_div.copy()
            logger.debug(f"[NOVELCOOL] copy() succeeded, type: {type(copied_div)}")
        except Exception as e:
            logger.error(f"[NOVELCOOL] copy() failed: {e}")
            return None
            
        try:
            cleaned_div = self._clean_content_container(copied_div)
            logger.debug(f"[NOVELCOOL] clean succeeded, type: {type(cleaned_div)}")
        except Exception as e:
            logger.error(f"[NOVELCOOL] clean failed: {e}")
            return None
```

#### Step 3: Systematically Test Each Hypothesis
- **Run minimal test**: Add logging and re-run to see which step fails
- **Check evidence**: Look at logs to see which hypothesis is supported
- **Falsify systematically**: Rule out hypotheses one by one

#### Step 4: Create General Solution Class
Once root cause is identified, create a solution that addresses the entire class of issues:

```python
def safe_content_extraction(self, selector_func, cleanup_func=None):
    """Generic safe extraction with comprehensive error handling."""
    try:
        element = selector_func()
        if not element:
            return None, "Element not found"
            
        if cleanup_func and hasattr(element, 'copy'):
            try:
                cleaned = cleanup_func(element.copy())
                return cleaned.get_text(strip=True), "Success"
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}, using raw extraction")
                return element.get_text(strip=True), "Cleanup failed, raw used"
        else:
            return element.get_text(strip=True), "Raw extraction"
            
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return None, f"Failed: {e}"
```

### Key Anti-Patterns to Avoid
❌ **Assumption-driven fixing**: "It's probably X, let me patch X"  
❌ **Single hypothesis testing**: Only considering one explanation  
❌ **Immediate patching**: Fixing before understanding  
❌ **Generic error handling**: `try/except: pass` without logging  

### Key Patterns to Follow  
✅ **Evidence-driven diagnosis**: Collect data first, then analyze  
✅ **Multiple hypothesis generation**: Consider uncorrelated explanations  
✅ **Systematic falsification**: Test and rule out hypotheses methodically  
✅ **Comprehensive logging**: Log enough data to understand state  
✅ **General solution design**: Fix the class of problems, not just the instance  

## RULES


1.Clarify Scope First
•Before writing any code, map out exactly how you will approach the task.
•Confirm your interpretation of the objective.
•Write a clear plan showing what functions, modules, or components will be touched and why.
•Do not begin implementation until this is done and reasoned through.

2.Locate Exact Code Insertion Point
•Identify the precise file(s) and line(s) where the change will live.
•Never make sweeping edits across unrelated files.
•If multiple files are needed, justify each inclusion explicitly.
•Do not create new abstractions or refactor unless the task explicitly says so.

3.Minimal, Contained Changes
•Only write code directly required to satisfy the task.
•Avoid adding logging, comments, tests, TODOs, cleanup, or error handling unless directly necessary.
•No speculative changes or “while we’re here” edits.
•All logic should be isolated to not break existing flows.

4.Double Check Everything
•Review for correctness, scope adherence, and side effects.
•Ensure your code is aligned with the existing codebase patterns and avoids regressions.
•Explicitly verify whether anything downstream will be impacted.

5.Deliver Clearly
•Summarize what was changed and why.
•List every file modified and what was done in each.
•If there are any assumptions or risks, flag them for review.

6.Maintain Organization
•Never create scripts, tests, or utilities in the project root directory.
•Use the organized folder structure: scripts/tests/, scripts/utils/, scripts/diagnostics/, scripts/examples/
•Keep temporary files in data/temp/ and exports in data/exports/
•Update scripts/README.md when adding new utility scripts.
•Remove obsolete test scripts regularly to avoid clutter.

## GIT COMMIT REQUIREMENTS

### Commit Message Format
All commits must follow this detailed format:

```
[TYPE]: Brief summary (50 chars max)

MOTIVATION:
- Why this change was needed
- What problem it solves
- Context/background

APPROACH:
- How the solution works
- Key technical decisions made
- Alternative approaches considered

CHANGES:
- file1.py: Specific changes made
- file2.py: Specific changes made
- etc.

IMPACT:
- What functionality is added/changed
- Any breaking changes
- Dependencies or setup changes needed

TESTING:
- How to verify the changes work
- Any manual testing performed
- Potential edge cases to watch

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Commit Types
- **feat**: New feature or major functionality
- **fix**: Bug fix or correction
- **refactor**: Code restructuring without changing functionality  
- **docs**: Documentation updates
- **chore**: Maintenance, dependencies, config changes
- **init**: Initial project setup

### Pre-Commit Requirements
1. **Always explain the plan** before making changes
2. **State the motivation** clearly in commit message
3. **List exact files** that will be modified and why
4. **Include testing steps** for the reviewer
5. **Backup critical files** before major changes

## ALIGNMENT MAP BUILDER - IMPLEMENTATION DETAILS (2025-07-15)

### ✅ COMPLETED IMPLEMENTATION (Phases 1-3)

#### **Phase 1: Enhanced Utils Module (`utils/alignment_map_builder.py`)**

**Key Functions Implemented:**
1. **`build_and_save_alignment_map(chinese_dir, english_dir, novel_name, output_path=None)`**
   - Complete workflow function with central storage
   - Returns: `(success, message, build_stats)`
   - Handles validation, building, and saving with backup

2. **`preview_alignment_mapping(chinese_dir, english_dir)`**
   - Preview system without actually building
   - Returns comprehensive statistics and validation results
   - Shows file issues and overlap analysis

3. **`validate_chapter_directories(chinese_dir, english_dir)`**
   - Comprehensive directory and file validation
   - Checks existence, permissions, file integrity
   - Returns detailed statistics and error reporting

4. **`scan_directory_for_chapters(directory_path, file_pattern=None, validate_files=True)`**
   - Directory-agnostic file scanning
   - Supports encoding detection (UTF-8, GBK, Latin-1)
   - Comprehensive file validation with size/content checks

5. **`get_alignment_map_path(novel_name)`**
   - Central storage path generator: `data/alignments/{novel}_alignment_map.json`
   - Automatic directory creation and filename sanitization

**Critical Technical Patterns:**
- **Fail-loud error handling** with categorical logging prefixes
- **Encoding detection** for Chinese text files
- **File validation** with size limits and content analysis
- **Central storage** in `data/alignments/` directory
- **Backup system** with timestamps before overwriting

#### **Phase 2: Data Review Page Integration (`pages/1_📖_Data_Review_Alignment.py`)**

**UI Integration Points:**
- **Directory selection UI** when no alignment map exists
- **Preview system** with statistics and file issue reporting
- **Progressive workflow**: Preview → Build → Results
- **Session state management** for UI flow control

**Key UI Components:**
```python
# Directory selection when no alignment map
if not alignment_map:
    st.header("🔨 Build Alignment Map")
    
    # Two-column directory selection
    chinese_dir = st.text_input("Chinese Directory Path:")
    english_dir = st.text_input("English Directory Path:")
    
    # Preview button triggers validation
    if st.button("👀 Preview Alignment"):
        preview_result = preview_alignment_mapping(chinese_dir, english_dir)
        # Display statistics and file issues
    
    # Build button triggers actual creation
    if st.button("🔨 Build Alignment Map"):
        success, message, stats = build_and_save_alignment_map(chinese_dir, english_dir, novel_name)
        # Clear session state and trigger rerun
```

#### **Phase 3: Home Dashboard Cleanup (`🏠_Home_Dashboard.py`)**

**Changes Made:**
- **Removed lines 120-182**: Entire Build Alignment Map section
- **Updated status checking** to check both central and legacy locations
- **Clean separation**: Home = scraping, Data Review = alignment building

**Status Checking Logic:**
```python
# Check central location first
alignments_dir = os.path.join("data", "alignments")
alignment_maps_found = len([f for f in os.listdir(alignments_dir) if f.endswith('_alignment_map.json')])

# Also check legacy location
legacy_alignment_exists = os.path.exists("alignment_map.json")
```

### 🔄 PENDING IMPLEMENTATION (Phases 4-5)

#### **Phase 4.1: Comprehensive Test Infrastructure** ⏳

**Required Test Structure:**
```
tests/
├── test_alignment_map_builder.py          # Unit tests
├── test_integration_alignment.py          # Integration tests
├── fixtures/                              # Test data
│   ├── chinese_chapters/                  # Sample Chinese files
│   ├── english_chapters/                  # Sample English files
│   ├── problematic_files/                 # Edge cases
│   └── expected_outputs/                  # Expected results
└── conftest.py                            # Pytest configuration
```

**Critical Test Cases:**
1. **Directory Validation Tests**
   - Non-existent directories
   - Permission errors
   - Empty directories
   - Mixed file types

2. **File Validation Tests**
   - Different encodings (UTF-8, GBK, Latin-1)
   - Corrupted files
   - HTML content instead of text
   - Extremely large files (>50MB)
   - Empty files

3. **Chapter Number Extraction Tests**
   - Various filename formats
   - Edge cases with special characters
   - Duplicate chapter numbers

4. **Alignment Map Building Tests**
   - Perfect overlap scenarios
   - Partial overlap scenarios
   - No overlap scenarios
   - Mixed valid/invalid files

5. **Central Storage Tests**
   - Path generation and sanitization
   - Backup creation and restoration
   - Concurrent access handling

**Test Data Requirements:**
- **Sample Chinese files** with proper encoding
- **Sample English files** with standard naming
- **Problematic files** for edge case testing
- **Expected alignment maps** for validation

**Key Test Functions to Implement:**
```python
def test_validate_chapter_directories_success():
    # Test successful validation with both directories
    
def test_validate_chapter_directories_missing():
    # Test handling of missing directories
    
def test_file_validation_encoding_detection():
    # Test automatic encoding detection
    
def test_build_alignment_map_perfect_overlap():
    # Test when Chinese and English chapters align perfectly
    
def test_central_storage_path_generation():
    # Test path generation and sanitization
```

#### **Phase 5.1: End-to-End Testing and Validation** ⏳

**Integration Test Scenarios:**
1. **Full Workflow Testing**
   - Start with two directories
   - Run preview
   - Build alignment map
   - Verify central storage
   - Test backup creation

2. **UI Integration Testing**
   - Simulate user directory selection
   - Test session state management
   - Verify page rerun behavior
   - Test error display

3. **Error Recovery Testing**
   - Handling of partial failures
   - Recovery from corrupted files
   - Graceful handling of permission errors

4. **Performance Testing**
   - Large directory handling (1000+ files)
   - Memory usage monitoring
   - Encoding detection performance

**Validation Checkpoints:**
- **Data integrity**: Verify alignment map accuracy
- **File preservation**: Ensure original files unchanged
- **Backup system**: Test backup creation and restoration
- **Error reporting**: Verify all failure modes logged properly
- **UI responsiveness**: Test with various directory sizes

### 🔧 IMPLEMENTATION GUIDELINES FOR PHASES 4-5

**Testing Infrastructure Setup:**
1. Create `tests/` directory structure
2. Install pytest dependencies: `pip install pytest pytest-mock`
3. Create test fixtures with known good/bad data
4. Implement unit tests for each function
5. Add integration tests for full workflows

**Key Testing Patterns:**
- **Mock file operations** for permission testing
- **Use temporary directories** for isolation
- **Parametrize tests** for different scenarios
- **Mock streamlit components** for UI testing

**Performance Benchmarks:**
- Directory scanning: <1s for 1000 files
- File validation: <5s for 1000 files
- Alignment map building: <10s for 1000 files
- Memory usage: <500MB for large datasets

**Error Handling Validation:**
- All error messages should be user-friendly
- All errors should be logged with context
- Recovery suggestions should be provided
- Partial failures should not corrupt data

## PLURALISTIC TRANSLATION WORKBENCH - FUTURE ROADMAP

### Current Implementation: MLOps Research Platform
**What We Built (Dec 2024):**
- 📖 **Data Review & Alignment**: Chapter-by-chapter curation with binary search misalignment detection
- 🤖 **Fine-tuning Workbench**: Complete model training pipeline with Google AI SDK
- 🧪 **Experimentation Lab**: Statistical model comparison and evaluation with BLEU/semantic scores
- 🔧 **Modular Architecture**: utils.py shared functions, multi-page Streamlit app

### Target Vision: Pluralistic Translation Workbench
**What We Want to Build:**

#### 🎨 **Prompt Engineering Suite** (New Page 4)
**Philosophy**: "There isn't one correct translation" - optimize for different purposes
- **Prompt Library**: Save/manage translation styles
  - "Literal": High accuracy, retain structure
  - "Dynamic": Modern Western audience, dramatic flow  
  - "Simplified": Young adult reading level
  - "Poetic": Emphasis on literary beauty
- **A/B Testing Interface**: Compare prompts on same chapter
- **Style Analytics**: Which prompts work best for different content types

#### 📊 **Enhanced Multi-Metric Dashboard** 
**Current**: BLEU + Semantic Similarity
**Target**: Complete evaluation suite
- **BLEU Score**: N-gram overlap, sentence structure
- **ROUGE Score**: Recall of key phrases  
- **BERT Semantic**: Meaning preservation (existing)
- **Readability Scores**: Flesch-Kincaid, SMOG
- **Style Consistency**: Terminology, tone analysis
- **Cultural Adaptation**: Idiom handling, localization quality

#### 📖 **EPUB Creator & Export Engine** (New Page 5)
**The "Productization" Step**
- **Translation Source Selection**: Choose style (Official, AI-Literal, AI-Dynamic, etc.)
- **Batch EPUB Generation**: Full novel in chosen translation style
- **Multi-Version Exports**: Generate multiple EPUB versions simultaneously
- **Quality Validation**: Pre-export consistency checking
- **Metadata Management**: Author, translator credits, version tracking

#### 🔀 **Unified Workflow Architecture**
```
Stage 1: Data Curation (existing) 
    → Perfect alignment_map.json

Stage 2: Model Training (existing)
    → Create specialized translation models  

Stage 3: Prompt Engineering (NEW)
    → Define translation styles and test approaches

Stage 4: Multi-Translation Generation (NEW)
    → Generate multiple versions of same content

Stage 5: Evaluation & Selection (enhanced)
    → Multi-metric comparison, human judgment integration

Stage 6: EPUB Export (NEW)
    → Production-ready translated novels
```

### Tradeoffs & Strategic Decisions

#### **Current MLOps Implementation vs Pluralistic Vision**

**MLOps Approach (What We Built)**
- ✅ **Audience**: ML researchers, academics
- ✅ **Output**: Trained models, research papers
- ✅ **Strengths**: Cutting-edge, publication-worthy, technical depth
- ❌ **Limitations**: High technical barrier, no immediate business value

**Pluralistic Approach (Original Vision)**  
- ✅ **Audience**: Content creators, publishers, translators
- ✅ **Output**: Custom translated books, multiple styles
- ✅ **Strengths**: Immediate business value, creative flexibility
- ❌ **Limitations**: Limited to existing model capabilities

#### **Recommended Hybrid Approach**
**Keep MLOps foundation AND add Pluralistic features**
- Serves both researchers AND content creators
- Unique market positioning (no one else has both)
- Technical depth + practical utility

### Implementation Priority

#### **Phase 1: Complete Pluralistic Vision (High Priority)**
1. **🎨 Prompt Engineering Suite**
   - Add pages/4_🎨_Prompt_Engineering.py
   - Prompt library with preset styles
   - A/B testing interface

2. **📊 Enhanced Evaluation Metrics**
   - Add ROUGE score calculation to utils.py
   - Implement readability metrics
   - Multi-metric comparison dashboard

3. **📖 EPUB Export Engine**
   - Add pages/5_📖_EPUB_Creator.py
   - Translation source selection
   - Batch export functionality

#### **Phase 2: Advanced Features (Medium Priority)**
1. **🎭 Style Analytics**: Which prompts work for which content
2. **🌍 Cultural Adaptation Tools**: Idiom and localization handling
3. **📝 Human-in-the-Loop Editing**: Post-AI translation refinement
4. **⚡ Performance Optimization**: Caching, parallel processing

#### **Phase 3: Business Features (Lower Priority)**  
1. **👥 Multi-User Support**: Team collaboration features
2. **🔐 Authentication & Permissions**: Role-based access
3. **☁️ Cloud Deployment**: Scale beyond local machine
4. **📈 Analytics Dashboard**: Usage tracking, quality trends
5. **🕷️ Dynamic Chapter Detection**: Auto-detect available chapters from target websites instead of hardcoded limits

### Known Issues to Debug
- **build_and_report.py**: Needs debugging for dataset export functionality
- **Dependencies**: Ensure all new requirements are properly specified
- **Error Handling**: Robust fallbacks for API failures and missing data

### File Structure After Full Implementation
```
/project_root
├── 📂 data/                    # All data, cache, and backups
├── 📂 pages/                   # Multi-page Streamlit app
│   ├── 2_🤖_Fine-tuning_Workbench.py
│   ├── 3_🧪_Experimentation_Lab.py  
│   ├── 4_🎨_Prompt_Engineering.py      # NEW
│   └── 5_📖_EPUB_Creator.py            # NEW
├── 📜 master_review_tool.py     # Home page (data curation)
├── 📜 utils.py                 # Shared functions
├── 📜 build_and_report.py      # Dataset export (to debug)
├── 📜 run_workbench.py         # Easy launcher
└── 📂 epub_exports/            # Generated EPUB files
```

### Success Metrics
**Technical Success**:
- All pages functional with error handling
- Multi-style translation generation working
- EPUB export producing valid files

**Business Success**:
- Content creators can generate custom translated novels
- Publishers can produce multiple reading-level versions  
- Researchers can develop and compare translation approaches

**Academic Success**:
- Framework suitable for publication
- Novel approach to translation style optimization
- Contribution to pluralistic translation research