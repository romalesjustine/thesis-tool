# LORECAST - News Article Summarizer

A web-based news article summarization tool powered by a fine-tuned Longformer Encoder-Decoder (LED) model. This thesis project provides intelligent text summarization with automatic category classification and dynamic parameter optimization.

## Table of Contents

- [Features](#features)
- [How the Model Works](#how-the-model-works)
- [System Requirements](#system-requirements)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)

## Features

- **Intelligent Summarization**: Uses a fine-tuned LED model for abstractive text summarization
- **Dynamic Optimization**: Automatically adjusts parameters based on article complexity
- **Category Classification**: Automatically categorizes articles (political, sports, business, technology, health, etc.)
- **Entity-Aware Processing**: Preserves important names, organizations, and numerical data
- **Hybrid Approach**: Combines extractive and abstractive summarization for optimal results
- **Real-time Processing**: Fast summarization with loading indicators

## How the Model Works

### Model Architecture

LORECAST uses **Longformer Encoder-Decoder (LED)**, a transformer-based model specifically designed for long documents:

- **Base Model**: `allenai/led-base-16384`
- **Fine-tuned Model Path**: `lorecast/model_final`
- **Max Input Length**: 4096 tokens (configurable up to 16384)
- **Summarization Type**: Abstractive (generates new text rather than extracting sentences)

### Summarization Pipeline

1. **Preprocessing Phase** (app.py:42-65)

   - Normalizes text (Unicode characters, spacing, punctuation)
   - Preserves numbers and important formatting
   - Cleans and standardizes input

2. **Content Analysis** (app.py:171-219)

   - Analyzes article complexity (length, entity density, data density)
   - Calculates complexity score (0-7 scale)
   - Determines optimal summarization parameters

3. **Hybrid Processing** (app.py:607-619)

   - **For articles > 500 words**: Uses extractive pre-filtering to identify key sentences
   - **For shorter articles**: Processes directly
   - Reduces computational load while preserving key information

4. **Entity & Number Extraction** (app.py:67-169)

   - Extracts key entities (people, organizations, locations)
   - Identifies important numbers with context (percentages, currency, statistics)
   - Preserves critical data points for inclusion in summary

5. **Dynamic Parameter Calculation** (app.py:221-276)

   - Adjusts summary length ratio (15-25% of original)
   - Sets beam search parameters (6-8 beams)
   - Configures length penalties based on complexity
   - Ensures proportional, accurate summaries

6. **Abstractive Generation** (app.py:460-472)

   - Uses beam search for high-quality output
   - Parameters:
     - `num_beams=11`: Explores multiple candidate summaries
     - `no_repeat_ngram_size=6`: Prevents repetition
     - `length_penalty=1.2`: Balances detail vs. brevity
     - `repetition_penalty=1.7`: Strong anti-repetition
     - `encoder_no_repeat_ngram_size=5`: Avoids copying input

7. **Category Classification** (app.py:330-389)
   - Keyword-based scoring across 13+ categories
   - Returns best-matching category with confidence threshold

## System Requirements

### Software Requirements

- **Python**: 3.8 or higher
- **Node.js**: 16.x or higher
- **npm**: 8.x or higher

### Hardware Requirements

- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 2GB for model files
- **GPU**: Optional (CUDA-compatible GPU for faster processing)
  - CPU inference works but is slower

### Operating System

- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 20.04+, Debian, etc.)

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd thesis-tool
```

### 2. Backend Setup (Python/Flask)

#### Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Download the Model

The model should be placed in the `lorecast/model_final` directory. Ensure you have:

- `config.json`
- `pytorch_model.bin` or `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

### 3. Frontend Setup (React/Vite)

#### Install Node Dependencies

```bash
npm install
```

#### Configure Proxy (Optional)

The `package.json` already includes a proxy configuration for the Flask backend:

```json
"proxy": "http://localhost:5000"
```

If needed, update `vite.config.js` for custom proxy settings.

## Running the Application

### Method 1: Run Backend and Frontend Separately

#### Terminal 1 - Start Flask Backend

```bash
# Activate virtual environment first
python app.py
```

The Flask server will start on `http://localhost:5000`

#### Terminal 2 - Start Vite Development Server

```bash
npm run dev
```

The frontend will start on `http://localhost:5173` (or next available port)

### Method 2: Production Build

#### Build Frontend

```bash
npm run build
```

#### Serve Production Build

```bash
npm run preview
```

## API Endpoints

### POST /api/summarize

Summarize an article.

**Request:**

```json
{
  "text": "Your article text here...",
  "mode": "smart"
}
```

**Response:**

```json
{
  "summary": "Generated summary...",
  "category": "Political"
}
```

**Modes**: `basic`, `smart` (default), `advanced`

### GET /api/health

Check API status.

**Response:**

```json
{
  "status": "healthy"
}
```

### GET /api/modes

Get available summarization modes.

**Response:**

```json
{
  "modes": {
    "basic": { "description": "...", "features": [...] },
    "smart": { "description": "...", "features": [...] },
    "advanced": { "description": "...", "features": [...] }
  },
  "default": "smart"
}
```

### GET /

API information and documentation.

## Project Structure

```
thesis-tool/
├── app.py                  # Flask backend with LED model
├── requirements.txt        # Python dependencies
├── package.json           # Node.js dependencies
├── vite.config.js         # Vite configuration
├── index.html             # HTML entry point
├── lorecast/              # Model files directory
│   └── model_final/       # Fine-tuned LED model
├── src/                   # React frontend source
│   ├── App.jsx           # Main React app
│   ├── Landing.jsx       # Landing page component
│   ├── Summarizer.jsx    # Summarizer UI component
│   ├── Howitworks.jsx    # Information page
│   └── styles/           # CSS stylesheets
└── README.md             # This file
```

## Model Architecture

### Longformer Encoder-Decoder (LED)

- **Attention Mechanism**: Sliding window + global attention
- **Context Window**: Handles up to 16,384 tokens
- **Parameters**: ~162M (base model)
- **Training**: Fine-tuned on news article dataset
- **Task**: Abstractive text summarization

### Key Technical Details

- **Tokenizer**: `LEDTokenizerFast` (fast Rust-based tokenizer)
- **Framework**: PyTorch + Hugging Face Transformers
- **Inference**: Beam search with dynamic parameters
- **Device**: Auto-detects CUDA GPU or falls back to CPU

### Performance Optimization

- Hybrid extractive-abstractive approach for long documents
- Dynamic parameter adjustment based on content
- Entity-aware prompting for better coverage
- Number preservation and restoration
- Hallucination detection and removal

## Troubleshooting

### Model Loading Issues

- Ensure the model path `lorecast/model_final` exists and contains all required files
- Check available RAM (model requires ~2GB)
- Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`

### CORS Errors

- Ensure Flask-CORS is installed: `pip install flask-cors`
- Check that the backend is running on port 5000
- Verify proxy configuration in `package.json` or `vite.config.js`

### Slow Inference

- Consider using a GPU if available
- Reduce `num_beams` in `app.py:706` (e.g., from 11 to 6)
- Reduce `max_new_tokens` for shorter summaries

### Port Conflicts

- Change Flask port in `app.py:944`: `app.run(port=5001)`
- Change Vite port in `vite.config.js`: `server: { port: 3000 }`

## Contributors

- Flores, Honey BSCS PUP Manila
- Gonzales, Grachella Jemyca BSCS PUP Manila
- Samaniego, Joshua BSCS PUP Manila
- Romales, Justine Carl BSCS PUP Manila

## Acknowledgments

- **Model**: Based on allenai/led-base-16384
- **Framework**: Hugging Face Transformers
- **Frontend**: React + Vite
- **Backend**: Flask + PyTorch
