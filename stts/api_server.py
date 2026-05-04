from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import tempfile
import os
import base64
import io
import uuid
import asyncio
from pathlib import Path

# Get the directory where this file is located
APP_DIR = Path(__file__).parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

# Setup templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Import SileroTTS with error handling
try:
    from stts.silero_tts import SileroTTS
    SILEROTTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing SileroTTS: {e}")
    SileroTTS = None
    SILEROTTS_AVAILABLE = False

app = FastAPI(
    title="SileroTTS API Server",
    version="0.8.2",
    description="Text-to-Speech API with audio streaming support"
)

# CORS middleware for global access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.mount("/static", name="static")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global instance to hold the current TTS instance
current_tts_instance = None

# Temp directory for audio files
AUDIO_DIR = Path(tempfile.gettempdir()) / "stts_audio"
AUDIO_DIR.mkdir(exist_ok=True)

# History metadata file
HISTORY_FILE = AUDIO_DIR / "history.json"

def load_history():
    """Load generation history from JSON file."""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading history: {e}")
    return []

def save_history(history):
    """Save generation history to JSON file."""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving history: {e}")

def add_to_history(text, language, model_id, speaker, sample_rate, duration, file_path):
    """Add a new generation to history."""
    history = load_history()
    
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "language": language,
        "model_id": model_id,
        "speaker": speaker,
        "sample_rate": sample_rate,
        "duration": duration,
        "filename": Path(file_path).name,
        "filepath": str(file_path)
    }
    
    history.insert(0, entry)
    
    # Keep only last 100 entries
    if len(history) > 100:
        history = history[:100]
    
    save_history(history)
    return entry

from datetime import datetime
import json


class ModelInfo(BaseModel):
    model_id: str
    language: str
    speaker: str
    sample_rate: int
    device: str
    put_accent: bool = True
    put_yo: bool = True
    num_threads: int = 6


class TTSRequest(BaseModel):
    text: str
    model_id: Optional[str] = None
    language: Optional[str] = None
    speaker: Optional[str] = None
    sample_rate: Optional[int] = None
    device: Optional[str] = "cpu"
    output_format: Optional[str] = "wav"
    autoplay: Optional[bool] = False


class TTSResponse(BaseModel):
    audio_data: str
    output_file: str
    duration: float
    sample_rate: int


class SettingsUpdate(BaseModel):
    model_id: Optional[str] = None
    language: Optional[str] = None
    speaker: Optional[str] = None
    sample_rate: Optional[int] = None
    device: Optional[str] = None
    put_accent: Optional[bool] = None
    put_yo: Optional[bool] = None
    num_threads: Optional[int] = None


def get_model_id_for_language(language: str) -> str:
    """Get default model ID for a language."""
    language_models = {
        "en": "v3_en",
        "ru": "v4_ru",
        "de": "thorsten_v2",
        "es": "tux_v2",
        "fr": "gilles_v2",
        "uk": "v4_uk",
        "uz": "v3_uz",
    }
    return language_models.get(language, "v4_ru")


def get_default_speaker(language: str) -> str:
    """Get default speaker for a language."""
    if language == "en":
        return "en_0"
    return "kseniya_v2"


@app.get("/ui")
async def get_ui():
    """Serve the modern web UI."""
    ui_file = TEMPLATES_DIR / "index.html"
    if ui_file.exists():
        return HTMLResponse(content=ui_file.read_text(encoding="utf-8"), media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="UI not found. Create templates/index.html")


def get_or_create_tts_instance(
    model_id: str = None,
    language: str = "ru",
    speaker: str = None,
    sample_rate: int = 48000,
    device: str = "cpu"
) -> SileroTTS:
    """Get or create TTS instance with specified parameters."""
    global current_tts_instance
    
    if not model_id:
        model_id = get_model_id_for_language(language)
    
    if not speaker:
        speaker = get_default_speaker(language)
    
    # Create new instance if needed
    if (current_tts_instance is None or
        model_id != current_tts_instance.model_id or
        language != current_tts_instance.language or
        speaker != current_tts_instance.speaker or
        sample_rate != current_tts_instance.sample_rate or
        device != current_tts_instance.device):
        
        current_tts_instance = SileroTTS(
            model_id=model_id,
            language=language,
            speaker=speaker,
            sample_rate=sample_rate,
            device=device
        )
    
    return current_tts_instance


def generate_audio_file(
    tts: SileroTTS,
    text: str,
    file_id: str = None
) -> str:
    """Generate audio file and return path."""
    if file_id is None:
        file_id = str(uuid.uuid4())
    
    output_path = AUDIO_DIR / f"tts_{file_id}.wav"
    result_file = tts.tts(str(text), str(output_path))
    return result_file


@app.get("/")
async def root(request: Request):
    """Root endpoint - serves the web UI."""
    return templates.TemplateResponse(
        "main.html",
        {"request": request}
    )
    

@app.get("/ui")
async def get_ui(request: Request):
    """Serve the modern web UI."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )
    
def default_speakers(lang: str):
    if lang == 'ru':
        model.speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']
    elif lang == 'en':
        model.speakers = ['en_0', 'en_1', 'en_2', 'en_3', 'en_4', 'en_5', 'en_6', 'en_7', 'en_8', 'en_9', 'en_10', 'random']
    else:
        model.speakers = ['random'] 

@app.get("/models")
async def get_models(request: Request, language: Optional[str] = None):
    """Get available models information - returns JSON or HTML."""
    if not SILEROTTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="SileroTTS library not available")
    
    try:
        models = SileroTTS.get_available_models()
        languages = SileroTTS.get_available_languages()
        
        # Get speakers for requested language or current instance
        speakers = {}
        sample_rates = {}
        speakers_list = []
        
        if language:
            # Create temporary instance to get speakers for requested language
            model_id = models[language][0] if models.get(language) else None
            if model_id:
                temp_tts = SileroTTS(
                    model_id=model_id,
                    language=language,
                    speaker=None,
                    sample_rate=48000,
                    device="cpu"
                )
                speaker_info = temp_tts.get_available_speakers()
                speakers[language] = speaker_info.get('speakers', []) if isinstance(speaker_info, dict) else speaker_info
                sample_rates[language] = SileroTTS.get_available_sample_rates_static(language, model_id)

                # Build speakers list for HTML template
                for spk in speakers[language]:
                    speakers_list.append({
                        'language': language,
                        'model_id': model_id,
                        'speaker': spk,
                        'sample_rates': sample_rates[language]
                    })
        elif current_tts_instance:
            print("elif cur")
            # Use current instance
            speaker_info = current_tts_instance.get_available_speakers()
            speakers[current_tts_instance.language] = speaker_info.get('speakers', []) if isinstance(speaker_info, dict) else speaker_info
            sample_rates[current_tts_instance.language] = {
                current_tts_instance.model_id: speaker_info.get('sample_rates', []) if isinstance(speaker_info, dict) else []
            }
        
            # Build speakers list for HTML template
            for spk in speakers[current_tts_instance.language]:
                speakers_list.append({
                    'language': current_tts_instance.language,
                    'model_id': current_tts_instance.model_id,
                    'speaker': spk,
                    'sample_rates': sample_rates[current_tts_instance.language].get(current_tts_instance.model_id, [])
                })
        
        current_model_info = {}
        if current_tts_instance:
            print("if curr")
            current_model_info = {
                "model_id": current_tts_instance.model_id,
                "language": current_tts_instance.language,
                "speaker": current_tts_instance.speaker,
                "sample_rate": current_tts_instance.sample_rate,
                "device": current_tts_instance.device
            }
        
        # Return HTML if browser accepts it
        accept_header = request.headers.get("accept", "")
        if "text/html" in accept_header:
            return templates.TemplateResponse(
                "models.html",
                {
                    "request": request,
                    "available_models": models,
                    "available_languages": languages,
                    "available_speakers": speakers_list,
                    "available_sample_rates": sample_rates
                }
            )
        
        return {
            "available_models": models,
            "available_languages": languages,
            "available_speakers": speakers,
            "available_sample_rates": sample_rates,
            "current_model": current_model_info,
            "speakers_list": speakers_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")


@app.get("/speakers")
async def get_speakers(language: str = Query(None, description="Language code"), model_id: Optional[str] = Query(None, description="Model ID")):
    """Get available speakers for a specific language/model."""
    if not SILEROTTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="SileroTTS library not available")
    
    try:
        models = SileroTTS.get_available_models()
        
        # Build comprehensive speakers list
        speakers_list = []
        
        # If language and model_id specified, return detailed info
        if language and model_id:
            if language not in models:
                raise HTTPException(status_code=400, detail=f"Language '{language}' not found")
            
            if model_id not in models[language]:
                raise HTTPException(status_code=400, detail=f"Model '{model_id}' not found for language '{language}'")
            
            loop = asyncio.get_event_loop()
            speaker_info = await loop.run_in_executor(None, lambda: _get_speakers_for_model(language, model_id))
            
            speakers_list.append({
                'language': language,
                'model_id': model_id,
                'speaker': speaker_info,
                'sample_rates': speaker_info.get('sample_rates', []) if isinstance(speaker_info, dict) else []
            })
            
            return {
                "language": language,
                "model_id": model_id,
                "speakers": speaker_info,
                "speakers_list": speakers_list
            }
        
        # Otherwise, return all speakers
        for lang, models_list in models.items():
            for m_id in models_list:
                try:
                    loop = asyncio.get_event_loop()
                    speaker_info = await loop.run_in_executor(None, lambda: _get_speakers_for_model(lang, m_id))
                    
                    speakers = speaker_info.get('speakers', []) if isinstance(speaker_info, dict) else speaker_info
                    sample_rates = speaker_info.get('sample_rates', []) if isinstance(speaker_info, dict) else []
                    
                    for speaker in speakers:
                        speakers_list.append({
                            'language': lang,
                            'model_id': m_id,
                            'speaker': speaker,
                            'sample_rates': sample_rates
                        })
                except Exception as e:
                    print(f"Error loading speakers for {lang}/{m_id}: {e}")
                    continue
        
        # Return HTML if requested
        return {
            "speakers_list": speakers_list
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting speakers: {str(e)}")


def _get_speakers_for_model(language: str, model_id: str):
    """Helper function to get speakers for a model (runs in thread pool)."""
    temp_tts = SileroTTS(
        model_id=model_id,
        language=language,
        speaker=None,
        sample_rate=48000,
        device="cpu"
    )
    
    speakers_list = temp_tts.get_available_speakers()
    sample_rates = temp_tts.get_available_sample_rates()
    
    return {
        "speakers": speakers_list,
        "sample_rates": sample_rates
    }


@app.post("/tts")
async def generate_tts_json(request: TTSRequest):
    """Generate speech from text - returns base64 encoded audio in JSON."""
    if not SILEROTTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="SileroTTS library not available")
    
    try:
        language = request.language or "ru"
        model_id = request.model_id or get_model_id_for_language(language)
        
        tts = get_or_create_tts_instance(
            model_id=model_id,
            language=language,
            speaker=request.speaker,
            sample_rate=request.sample_rate or 48000,
            device=request.device or "cpu"
        )
        
        # Run TTS generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        file_id = str(uuid.uuid4())
        output_path = AUDIO_DIR / f"tts_{file_id}.wav"
        
        result_file = await loop.run_in_executor(None, lambda: tts.tts(request.text, str(output_path)))
        
        with open(result_file, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
        file_stats = os.stat(result_file)
        duration = file_stats.st_size / (tts.sample_rate * 2)
        
        return TTSResponse(
            audio_data=audio_data,
            output_file=result_file,
            duration=duration,
            sample_rate=tts.sample_rate
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating TTS: {str(e)}")


@app.post("/tts/audio")
async def generate_tts_audio(request: TTSRequest):
    """
    Generate speech from text - returns audio file directly.
    Use this endpoint for global access and direct audio playback.
    """
    if not SILEROTTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="SileroTTS library not available")
    
    try:
        language = request.language or "ru"
        model_id = request.model_id or get_model_id_for_language(language)
        
        tts = get_or_create_tts_instance(
            model_id=model_id,
            language=language,
            speaker=request.speaker,
            sample_rate=request.sample_rate or 48000,
            device=request.device or "cpu"
        )
        
        # Run TTS generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        file_id = str(uuid.uuid4())
        output_path = AUDIO_DIR / f"tts_{file_id}.wav"
        
        result_file = await loop.run_in_executor(None, lambda: tts.tts(request.text, str(output_path), autoplay=request.autoplay))
        
        # Calculate duration
        file_stats = os.stat(result_file)
        duration = file_stats.st_size / (tts.sample_rate * 2)
        
        # Add to history
        add_to_history(
            text=request.text,
            language=language,
            model_id=model_id,
            speaker=request.speaker,
            sample_rate=tts.sample_rate,
            duration=duration,
            file_path=result_file
        )
        
        # Generate filename for download
        filename = f"tts_{language}_{file_id[:8]}.wav"
        
        return FileResponse(
            path=result_file,
            media_type="audio/wav",
            filename=filename
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating TTS: {str(e)}")


@app.post("/tts/stream")
async def generate_tts_stream(request: TTSRequest):
    """
    Generate speech from text - streams audio response.
    Best for real-time playback and reduces latency.
    """
    if not SILEROTTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="SileroTTS library not available")
    
    try:
        language = request.language or "ru"
        model_id = request.model_id or get_model_id_for_language(language)
        
        tts = get_or_create_tts_instance(
            model_id=model_id,
            language=language,
            speaker=request.speaker,
            sample_rate=request.sample_rate or 48000,
            device=request.device or "cpu"
        )
        
        # Run TTS generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        file_id = str(uuid.uuid4())
        output_path = AUDIO_DIR / f"tts_{file_id}.wav"
        
        result_file = await loop.run_in_executor(None, lambda: tts.tts(request.text, str(output_path), autoplay=request.autoplay))
        
        # Stream the file
        async def iter_file():
            loop = asyncio.get_event_loop()
            def read_chunks():
                with open(result_file, 'rb') as f:
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        yield chunk
            # Convert sync generator to async
            for chunk in read_chunks():
                yield chunk
        
        filename = f"tts_{language}_{file_id[:8]}.wav"
        
        return StreamingResponse(
            iter_file(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"inline; filename={filename}",
                "Accept-Ranges": "bytes"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating TTS: {str(e)}")


# Query-based endpoints for easy browser testing
@app.get("/tts/audio")
async def generate_tts_audio_get(
    text: str = Query(..., description="Text to convert to speech"),
    language: str = Query("ru", description="Language code"),
    model_id: str = Query(None, description="Model ID"),
    speaker: str = Query(None, description="Speaker name"),
    sample_rate: int = Query(48000, description="Sample rate"),
    device: str = Query("cpu", description="Device (cpu/cuda)")
):
    """GET version of /tts/audio for browser testing."""
    request = TTSRequest(
        text=text,
        language=language,
        model_id=model_id,
        speaker=speaker,
        sample_rate=sample_rate,
        device=device
    )
    return await generate_tts_audio(request)


@app.get("/tts/stream")
async def generate_tts_stream_get(
    text: str = Query(..., description="Text to convert to speech"),
    language: str = Query("ru", description="Language code"),
    model_id: str = Query(None, description="Model ID"),
    speaker: str = Query(None, description="Speaker name"),
    sample_rate: int = Query(48000, description="Sample rate"),
    device: str = Query("cpu", description="Device (cpu/cuda)")
):
    """GET version of /tts/stream for browser testing."""
    request = TTSRequest(
        text=text,
        language=language,
        model_id=model_id,
        speaker=speaker,
        sample_rate=sample_rate,
        device=device
    )
    return await generate_tts_stream(request)


@app.get("/settings")
async def get_settings():
    """Get current settings of the TTS instance."""
    if not SILEROTTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="SileroTTS library not available")
    
    try:
        if current_tts_instance:
            return {
                "model_id": current_tts_instance.model_id,
                "language": current_tts_instance.language,
                "speaker": current_tts_instance.speaker,
                "sample_rate": current_tts_instance.sample_rate,
                "device": current_tts_instance.device,
                "put_accent": current_tts_instance.put_accent,
                "put_yo": current_tts_instance.put_yo,
                "num_threads": current_tts_instance.num_threads
            }
        else:
            return {
                "error": "No TTS instance is currently initialized. Please make a TTS request first."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting settings: {str(e)}")


@app.post("/settings")
async def update_settings(settings: SettingsUpdate):
    """Update settings of the TTS instance."""
    if not SILEROTTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="SileroTTS library not available")
    
    global current_tts_instance
    
    try:
        if current_tts_instance is None:
            current_tts_instance = SileroTTS(
                model_id="v4_ru",
                language="ru",
                speaker="kseniya_v2",
                sample_rate=48000,
                device="cpu"
            )
        
        if settings.model_id is not None:
            current_tts_instance.change_model(settings.model_id)
        if settings.language is not None:
            current_tts_instance.change_language(settings.language)
        if settings.speaker is not None:
            current_tts_instance.change_speaker(settings.speaker)
        if settings.sample_rate is not None:
            current_tts_instance.change_sample_rate(settings.sample_rate)
        if settings.device is not None:
            current_tts_instance.device = settings.device
        if settings.put_accent is not None:
            current_tts_instance.put_accent = settings.put_accent
        if settings.put_yo is not None:
            current_tts_instance.put_yo = settings.put_yo
        if settings.num_threads is not None:
            current_tts_instance.num_threads = settings.num_threads
            import torch
            torch.set_num_threads(settings.num_threads)
        
        return {
            "status": "Settings updated successfully",
            "updated_settings": {
                "model_id": current_tts_instance.model_id,
                "language": current_tts_instance.language,
                "speaker": current_tts_instance.speaker,
                "sample_rate": current_tts_instance.sample_rate,
                "device": current_tts_instance.device,
                "put_accent": current_tts_instance.put_accent,
                "put_yo": current_tts_instance.put_yo,
                "num_threads": current_tts_instance.num_threads
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating settings: {str(e)}")


@app.get("/history")
async def get_history(request: Request):
    """Get generation history with audio file URLs."""
    try:
        history = load_history()
        
        # Build full URLs for audio files
        base_url = f"{request.url.scheme}://{request.url.netloc}"
        for entry in history:
            entry['audio_url'] = f"{base_url}/audio/{entry['filename']}"
        
        return {"history": history, "total": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")


@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve audio file from cache."""
    file_path = AUDIO_DIR / filename
    if file_path.exists():
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            filename=filename
        )
    raise HTTPException(status_code=404, detail="Audio file not found")


@app.delete("/cache")
async def clear_cache():
    """Clear temporary audio cache and history."""
    try:
        count = 0
        for file in AUDIO_DIR.glob("tts_*.wav"):
            file.unlink()
            count += 1
        
        # Clear history file
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        
        return {"status": "Cache cleared", "files_deleted": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


@app.get("/cache")
async def get_cache_info():
    """Get information about cached audio files."""
    try:
        files = list(AUDIO_DIR.glob("tts_*.wav"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "files_count": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache info: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")