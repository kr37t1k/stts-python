# SileroTTS - Stable Release

**README is available in the following languages:**

[![EN](https://img.shields.io/badge/EN-blue.svg)](https://github.com/kr37t1k/stts-python)
[![RU](https://img.shields.io/badge/RU-red.svg)](https://github.com/kr37t1k/stts-python/blob/main/README.RU.MD)

Silero TTS is a Python library that provides an easy way to synthesize speech from text using various Silero TTS models, languages, and speakers. It can be used as a standalone script or integrated into your own Python projects.

## Features

- Support for multiple languages and models
- Automatic downloading of the latest model configuration file with retry logic
- Text preprocessing and transliteration
- Batch processing of text files
- Detailed logging with loguru
- Progress tracking with tqdm
- Customizable options for sample rate, device, and more
- Can be used as a standalone script or integrated into Python code
- Robust error handling and validation
- Network resilience with retry mechanisms

### API Server Features

- **Web UI** - Modern responsive interface for TTS generation
- **REST API** - Full REST API with multiple endpoints
- **Streaming Support** - Real-time audio streaming
- **History & Cache** - Automatic audio file caching and generation history
- **Analytics Dashboard** - Comprehensive server monitoring and reachability analysis
  - Real-time metrics (uptime, response time, success rate)
  - Interactive charts and graphs
  - Endpoint performance monitoring
  - Error analysis and timeline
  - Export/Import session data (JSON/CSV)
  - Auto-refresh every 30 seconds

## Installation

### git+build
   ```bash
   git clone https://github.com/kr37t1k/stts-python
   python -m build
   python -m pip install "dist/silerotts-VTag-py3-none-any.whl"
   ```

### By pip within git:
   ```bash
   python -m pip install "git+https://github.com/kr37t1k/stts-python"
   ```

### By PyPI (simpler)
   ```bash
   python -m pip install stts-python
   ```

## Usage

### As a Standalone Script

You can use Silero TTS as a standalone script to synthesize speech from text files or directories containing text files.

```
python -m stts [options]
```

#### Options

- `--list-models`: List available models
- `--list-speakers`: List available speakers for a model
- `--language LANGUAGE`: Specify the language code (required)
- `--model MODEL`: Specify the model ID (default: latest version for the language)
- `--speaker SPEAKER`: Specify the speaker name (default: first available speaker for the model)
- `--sample-rate SAMPLE_RATE`: Specify the sample rate (default: 48000)
- `--device DEVICE`: Specify the device to use (default: cpu)
- `--text TEXT`: Specify the text to synthesize
- `--input-file INPUT_FILE`: Specify the input text file to synthesize
- `--input-dir INPUT_DIR`: Specify the input directory with text files to synthesize
- `--output-file OUTPUT_FILE`: Specify the output audio file (default: output.wav)
- `--output-dir OUTPUT_DIR`: Specify the output directory for synthesized audio files (default: output)
- `--log-level INFO` : Specify log-level, you can turn off use NONE value (default: INFO)

#### Examples

1. Synthesize speech from a text:
   ```
   python -m stts --language ru --text "Привет, мир!"
   ```

2. Synthesize speech from a text file:
   ```
   python -m stts --language en --input-file input.txt --output-file output.wav
   ```

3. Synthesize speech from multiple text files in a directory:
   ```
   python -m stts --language es --input-dir texts --output-dir audio
   ```

### As a Python Library

You can also integrate Silero TTS into your own Python projects by importing the `SileroTTS` class and using its methods.

```python
from stts.silero_tts import SileroTTS

# Get available models
models = SileroTTS.get_available_models()
print("Available models:", models)

# Get available languages
languages = SileroTTS.get_available_languages()
print("Available languages:", languages)

# Get the latest model for a specific language
latest_model = SileroTTS.get_latest_model('ru')
print("Latest model for Russian:", latest_model)

# Get available sample rates for a specific model and language
sample_rates = SileroTTS.get_available_sample_rates_static('ru', latest_model)
print("Available sample rates for the latest Russian model:", sample_rates)

# Initialize the TTS object
tts = SileroTTS(model_id='v3_en', language='en', speaker='en_2', sample_rate=48000, device='cpu')

# Synthesize speech from text
text = "Hello world!"
tts.tts(text, 'output.wav')

# Synthesize speech from a text file
# tts.from_file('input.txt', 'output.wav')

# Get available speakers for the current model
speakers = tts.get_available_speakers()
print("Available speakers for the current model:", speakers)

# Change the language
tts.change_language('en')
print("Language changed to:", tts.language)
print("New model ID:", tts.model_id)
print("New available speakers:", tts.get_available_speakers())

# Change the model
tts.change_model('v3_en')
print("Model changed to:", tts.model_id)
print("New available speakers:", tts.get_available_speakers())

# Change the speaker
tts.change_speaker('en_0')
print("Speaker changed to:", tts.speaker)

# Change the sample rate
tts.change_sample_rate(24000)
print("Sample rate changed to:", tts.sample_rate)
```

## CLI Features

The Silero TTS CLI provides the following features:

- **Language Support**: Specify the language code using the `--language` flag to synthesize speech in the desired language.
- **Model Selection**: Choose a specific model using the `--model` flag or let the CLI automatically select the latest model for the specified language.
- **Speaker Selection**: Select a speaker using the `--speaker` flag or use the default speaker for the chosen model.
- **Sample Rate**: Customize the sample rate of the synthesized speech using the `--sample-rate` flag.
- **Device**: Specify the device (CPU or GPU) to use for synthesis using the `--device` flag.
- **Text Input**: Provide the text to synthesize directly using the `--text` flag or specify an input text file using the `--input-file` flag.
- **Batch Processing**: Process multiple text files in a directory using the `--input-dir` flag.
- **Output**: Specify the output audio file using the `--output-file` flag or the output directory for batch processing using the `--output-dir` flag.
- **Model Listing**: List all available models using the `--list-models` flag.
- **Speaker Listing**: List all available speakers for a specific model using the `--list-speakers` flag.
- **Error Handling**: Comprehensive error handling with retry mechanisms for network operations.

## API Server

The SileroTTS API Server provides a web-based interface and REST API for text-to-speech generation.

1. Run the server
   ```batch
   python -m stts --server
   ```
   
2. Access the web UI:
   - Main UI: http://localhost:8002
   - Models Info: http://localhost:8002/models
   - Analytics Dashboard: http://localhost:8002/data

### API Endpoints

#### TTS Generation
- `POST /tts` - Generate audio (returns base64 encoded JSON)
- `POST /tts/audio` - Generate audio (returns audio file)
- `POST /tts/stream` - Stream audio response
- `GET /tts/audio` - GET version for browser testing
- `GET /tts/stream` - GET version for browser testing

#### Management
- `GET /models` - Get available models and speakers
- `GET /speakers` - Get available speakers for language/model
- `GET /settings` - Get current TTS settings
- `POST /settings` - Update TTS settings
- `GET /history` - Get generation history
- `DELETE /cache` - Clear audio cache
- `GET /cache` - Get cache information

### Analytics API Endpoints

#### Analytics Data
- `GET /api/analytics` - Get all analytics data
- `GET /api/analytics/export?format=json|csv&limit=1000` - Export data (JSON/CSV)
- `POST /api/analytics/import` - Import session data
- `DELETE /api/analytics` - Clear analytics
- `POST /api/analytics/check` - Record manual reachability check

#### Health & Monitoring
- `GET /api/health` - Basic health check (status, health_score)
- `GET /api/health?detailed=true` - Detailed health check with metrics
- `GET /api/metrics` - Prometheus-style metrics for monitoring
- `GET /api/system/info` - System information (CPU, memory, disk, network)

#### Performance & Statistics
- `GET /api/performance` - Detailed performance statistics
- `GET /api/endpoints` - Status of all tracked endpoints
- `GET /api/history?limit=100&endpoint=/tts/audio` - Request history with filtering
- `GET /api/stats/hourly?hours=24` - Hourly aggregated statistics

### OpenAI-Compatible Endpoints

#### Chat Completions (TTS wrapped in chat format)
- `POST /v1/chat/completions` - Generate speech in chat completion format
- `GET /v1/voices` - List available voices/speakers
- `GET /v1/models` - List available models with speakers (OpenAI-compatible)

```bash
# Chat completion example:
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "v4_ru_kseniya_v2",
    "messages": [
      {"role": "user", "content": "Привет, как дела?"}
    ],
    "stream": false
  }'
```

#### Voices List
```bash
# Get all available voices:
curl http://localhost:8002/v1/voices
```

### Usage with CherryStudio/Jan

1. **Add Custom Model**:
   - Open CherryStudio/Jan settings
   - Add custom API endpoint: `http://localhost:8002`
   - Select TTS mode
   - Choose voice from list (e.g., `v4_ru_kseniya_v2`)

2. **Configuration**:
   ```
   API Base URL: http://localhost:8002
   API Version: v1
   Model: v4_ru_kseniya_v2 (or any available voice)
   ```

3. **Supported Clients**:
   - ✅ CherryStudio
   - ✅ Jan
   - ✅ Open WebUI
   - ✅ AnythingLLM
   - ✅ LM Studio (with configuration)
   - ✅ OpenAI-compatible apps

### Analytics Dashboard

The analytics dashboard provides comprehensive monitoring of server performance and reachability:

**Key Features:**
- Real-time metrics (uptime, response time, success rate)
- System load monitoring (CPU, memory)
- Response time trends and distribution
- Endpoint performance tracking
- Error analysis and timeline
- Request volume statistics
- Export/Import session data
- Auto-refresh every 30 seconds

**Metrics Tracked:**
- Uptime percentage
- Average response time
- Success/error rates
- CPU and memory usage
- Request rates
- Per-endpoint statistics

**Data Export:**
- JSON format for full data preservation
- CSV format for spreadsheet analysis
- Session import for comparison
- Local storage in browser

### Example API Usage

```python
import requests

# Generate speech
response = requests.post(
    'http://localhost:8002/tts/audio',
    json={
        'text': 'Hello, world!',
        'language': 'en'
    }
)

# Save audio file
with open('output.wav', 'wb') as f:
    f.write(response.content)

# Get analytics
analytics = requests.get('http://localhost:8002/api/analytics').json()
print(f"Uptime: {analytics['metrics']['uptime']}%")
print(f"Avg Response: {analytics['metrics']['avgResponseTime']}ms")
```

### Streaming Example

```python
import requests

# Stream audio
response = requests.post(
    'http://localhost:8002/tts/stream',
    json={
        'text': 'Hello, world!',
        'language': 'en'
    },
    stream=True
)

# Play audio in real-time
with open('output.wav', 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

## Supported Languages

- Russian (ru)
- English (en)
- German (de)
- Spanish (es)
- French (fr)
- Bashkir (ba)
- Kalmyk (xal)
- Tatar (tt)
- Uzbek (uz)
- Ukrainian (ua)
- Indic (indic)
- Cyrillic (cyrillic)

## Stability Features (v0.6)

- Input validation for all parameters
- Network resilience with retry mechanisms and timeouts
- Graceful error handling with informative messages
- Proper resource management
- Comprehensive logging for debugging
- KeyboardInterrupt handling in CLI
- Robust model loading with fallbacks

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Silero Models](https://github.com/snakers4/silero-models) big thanks for providing the free TTS models
- [silero_tts_standalone](https://github.com/S-trace/silero_tts_standalone) this library inspired daswer123 to create his (daswer123/silero-tts-enhanced) project. We made fork from his project and remade it for own.