from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import tempfile
import os
import base64
import io
import uuid

# Import SileroTTS with error handling
try:
    from stts.silero_tts import SileroTTS
    SILEROTTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing SileroTTS: {e}")
    SileroTTS = None
    SILEROTTS_AVAILABLE = False

app = FastAPI(title="SileroTTS API Server", version="1.0.0")

# Global instance to hold the current TTS instance
current_tts_instance = None


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
    output_format: Optional[str] = "wav"  # Format of the output audio


class TTSResponse(BaseModel):
    audio_data: str  # Base64 encoded audio data
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


@app.get("/models")
async def get_models():
    """Get available models information"""
    if not SILEROTTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="SileroTTS library not available")
    
    try:
        models = SileroTTS.get_available_models()
        languages = SileroTTS.get_available_languages()
        
        # Get current model info if there's an active instance
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
async def generate_tts(request: TTSRequest):
    """Generate speech from text using the TTS model"""
    if not SILEROTTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="SileroTTS library not available")
    
    global current_tts_instance
    
    try:
        # Determine the model_id based on the language if not provided
        model_id = request.model_id
        language = request.language or "ru"
        
        # If no model_id is provided, select a default model based on language
        if not model_id:
            if language == "en":
                model_id = "lj_v2"  # English model
            elif language == "ru":
                model_id = "v4_ru"  # Russian model
            elif language == "de":
                model_id = "thorsten_v2"  # German model
            elif language == "es":
                model_id = "tux_v2"  # Spanish model
            elif language == "fr":
                model_id = "gilles_v2"  # French model
            else:
                # Default to Russian model if language is not recognized
                model_id = "v4_ru"
        
        # If no current instance exists, create one with defaults
        if current_tts_instance is None:
            current_tts_instance = SileroTTS(
                model_id=model_id,
                language=language,
                speaker=request.speaker or "en_0" if language == "en" else "kseniya_v2",
                sample_rate=request.sample_rate or 48000,
                device=request.device or "cpu"
            )
        
        # If request specifies different parameters, create new instance
        if (model_id != current_tts_instance.model_id or
            language != current_tts_instance.language or
            (request.speaker and request.speaker != current_tts_instance.speaker) or
            (request.sample_rate and request.sample_rate != current_tts_instance.sample_rate) or
            (request.device and request.device != current_tts_instance.device)):
            
            current_tts_instance = SileroTTS(
                model_id=model_id,
                language=language,
                speaker=request.speaker or current_tts_instance.speaker,
                sample_rate=request.sample_rate or current_tts_instance.sample_rate,
                device=request.device or current_tts_instance.device
            )
        
        # Generate temporary file for output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            output_path = temp_file.name
        
        # Generate speech
        result_file = current_tts_instance.tts(request.text, output_path)
        
        # Read the generated audio file and encode it as base64
        with open(result_file, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
        # Get file size to calculate duration (assuming 16-bit, mono)
        file_stats = os.stat(result_file)
        duration = file_stats.st_size / (current_tts_instance.sample_rate * 2)  # 2 bytes per sample for 16-bit
        
        # Clean up temporary file
        if os.path.exists(result_file):
            os.remove(result_file)
        
        return TTSResponse(
            audio_data=audio_data,
            output_file=result_file,
            duration=duration,
            sample_rate=current_tts_instance.sample_rate
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating TTS: {str(e)}")


@app.get("/settings")
async def get_settings():
    """Get current settings of the TTS instance"""
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
    """Update settings of the TTS instance"""
    if not SILEROTTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="SileroTTS library not available")
    
    global current_tts_instance
    
    try:
        if current_tts_instance is None:
            # If no instance exists, create a default one
            current_tts_instance = SileroTTS(
                model_id="v4_ru",
                language="ru",
                speaker="kseniya_v2",
                sample_rate=48000,
                device="cpu"
            )
        
        # Update the instance with provided settings
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
            
            # If number of threads changed, update torch
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


@app.get("/")
async def root():
    """Root endpoint to check if server is running"""
    return {"message": "SileroTTS API Server is running!", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8002, log_level="info")