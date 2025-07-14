# Translation Framework Workbench Setup

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key** (Choose one option)

   **Option A: Environment Variable (Recommended)**
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

   **Option B: Configuration File**
   ```bash
   cp config.example.json config.json
   # Edit config.json with your API key
   ```

3. **Launch Workbench**
   ```bash
   python run_workbench.py
   ```
   
   **Alternative direct launch:**
   ```bash
   streamlit run app.py
   ```

## API Key Setup

### Get Your Gemini API Key
1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Create a new API key
4. Copy the key for configuration

### Configuration Options

#### Environment Variable (Recommended)
```bash
# Linux/Mac
export GEMINI_API_KEY="your-api-key-here"

# Windows
set GEMINI_API_KEY=your-api-key-here
```

#### Configuration File
```bash
# Copy template
cp config.example.json config.json

# Edit config.json
{
  "gemini_api_key": "your-actual-api-key-here",
  "default_model": "gemini-2.5-pro",
  "default_history_count": 5,
  "api_delay": 1.0
}
```

## Configuration Priority

The system loads configuration in this order:
1. **Environment Variable** (`GEMINI_API_KEY`) - highest priority
2. **Config File** (`config.json`) - fallback
3. **Error** if neither is available

## Security Notes

- ✅ `config.json` is automatically ignored by git
- ✅ `config.example.json` is committed (without real keys)
- ✅ Environment variables are not stored in files
- ⚠️ Never commit real API keys to version control

## Troubleshooting

### "API Key not configured" Error
- Check environment variable: `echo $GEMINI_API_KEY`
- Verify config.json exists and has valid JSON
- Ensure key is properly quoted in config.json

### Config File Issues
```bash
# Validate JSON syntax
python -m json.tool config.json

# Check file exists
ls -la config.json
```

### Environment Variable Issues
```bash
# Set for current session
export GEMINI_API_KEY="your-key"

# Add to shell profile for persistence
echo 'export GEMINI_API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc
```

## Workbench Pages

Once configured, you'll have access to:

1. **🏠 Home** - Main dashboard with system status and quick navigation
2. **📖 Data Review & Alignment** - Dataset curation and quality control
3. **🤖 Fine-tuning Workbench** - Model training and management
4. **🧪 Pluralistic Translation Lab** - Multi-style translation generation
5. **📈 Experimentation Analysis** - Quality comparison and evaluation

**Navigation**: Use the **sidebar navigation** (left side) to switch between pages. The multi-page interface provides seamless access to all workbench tools.