# Scripts Directory

This directory contains utility, test, and diagnostic scripts organized by purpose.

## 📁 Directory Structure

```
scripts/
├── tests/           # Test scripts and small validation tools
├── utils/           # Utility scripts for data processing
├── diagnostics/     # Debugging and analysis tools
└── examples/        # Example usage and demo scripts
```

## 📋 Current Scripts

### 🧪 Tests (`scripts/tests/`)
- `test_build_with_args.py` - Demo of improved build_and_report.py features
- `test_build_small.py` - Creates minimal alignment map for testing

### 🔍 Diagnostics (`scripts/diagnostics/`)
- `diagnose_alignment.py` - Analyzes alignment map vs actual files

### 🛠️ Utils (`scripts/utils/`)
- (Utility scripts will be added here)

### 📚 Examples (`scripts/examples/`)
- (Example scripts will be added here)

## 🚀 Usage

Run scripts from the project root directory:

```bash
# Run diagnostic tools
python scripts/diagnostics/diagnose_alignment.py

# Run tests
python scripts/tests/test_build_small.py

# Run utilities
python scripts/utils/script_name.py
```

## 📝 Guidelines

### For Test Scripts
- Prefix with `test_`
- Include brief description and usage examples
- Should be self-contained and not modify production data

### For Utility Scripts
- Use descriptive names (e.g., `fix_alignment_map.py`)
- Include help text and examples
- Add error handling and validation

### For Diagnostic Scripts
- Prefix with `diagnose_` or `analyze_`
- Should be read-only and safe to run
- Provide clear output and recommendations

## 🧹 Maintenance

- Keep this README updated when adding new scripts
- Remove obsolete scripts regularly
- Consider moving frequently-used utilities to main directory