from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
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
    version="0.7",
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

# Global instance to hold the current TTS instance
current_tts_instance = None

# Temp directory for audio files
AUDIO_DIR = Path(tempfile.gettempdir()) / "stts_audio"
AUDIO_DIR.mkdir(exist_ok=True)


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
async def root():
    """Root endpoint to check if server is running."""
    return {
        "message": "SileroTTS API Server v2.0 is running!",
        "ui": "/ui",
        "docs": "/docs",
        "endpoints": {
            "POST /tts": "Generate TTS - returns base64 audio (JSON)",
            "POST /tts/audio": "Generate TTS - returns audio file directly",
            "POST /tts/stream": "Generate TTS - streams audio response",
            "GET /tts/audio": "GET version for browser testing",
            "GET /tts/stream": "GET version for browser testing",
            "GET /models": "List available models",
            "GET /settings": "Get current settings",
            "POST /settings": "Update settings",
            "DELETE /cache": "Clear temporary audio cache"
        },
        "version": "2.0.0"
    }


@app.get("/models")
async def get_models():
    """Get available models information."""
    if not SILEROTTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="SileroTTS library not available")
    
    try:
        models = SileroTTS.get_available_models()
        languages = SileroTTS.get_available_languages()
        
        current_model_info = {}
        if current_tts_instance:
            current_model_info = {
                "model_id": current_tts_instance.model_id,
                "language": current_tts_instance.language,
                "speaker": current_tts_instance.speaker,
                "sample_rate": current_tts_instance.sample_rate,
                "device": current_tts_instance.device
            }
        
        return {
            "available_models": models,
            "available_languages": languages,
            "current_model": current_model_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")


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
        
        file_id = str(uuid.uuid4())
        output_path = AUDIO_DIR / f"tts_{file_id}.wav"
        
        result_file = tts.tts(request.text, str(output_path))
        
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
        
        file_id = str(uuid.uuid4())
        output_path = AUDIO_DIR / f"tts_{file_id}.wav"
        
        result_file = tts.tts(request.text, str(output_path))
        
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
        
        file_id = str(uuid.uuid4())
        output_path = AUDIO_DIR / f"tts_{file_id}.wav"
        
        result_file = tts.tts(request.text, str(output_path))
        
        # Stream the file
        def iter_file():
            with open(result_file, 'rb') as f:
                while chunk := f.read(8192):
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


@app.delete("/cache")
async def clear_cache():
    """Clear temporary audio cache."""
    try:
        count = 0
        for file in AUDIO_DIR.glob("tts_*.wav"):
            file.unlink()
            count += 1
        return {"status": "Cache cleared", "files_deleted": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")