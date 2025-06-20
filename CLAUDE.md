# Claude Code Configuration

## Project: Way of the Devil Translation System

### CURRENT STATUS (2025-06-20)
**MAJOR PROGRESS ACHIEVED:**
- ✅ Raw data scraping working perfectly (dxmwx.org)
- ✅ English EPUB processing complete (772 chapters extracted)
- ✅ Reconnaissance and framework validation complete
- 🔄 **CURRENT TASK**: Align first 772 Chinese raw chapters with English translation

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