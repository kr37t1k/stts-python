from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, AsyncGenerator
import tempfile
import os
import base64
import io
import uuid
import asyncio
import time
import json
import platform
import socket
import psutil
import yaml
import torch
from pathlib import Path
from collections import deque
from statistics import mean, stdev

# Get the directory where this file is located
APP_DIR = Path(__file__).parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

# Setup templates
TEMPLATES = {}
try:
    from jinja2 import Environment, FileSystemLoader
    template_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    
    def get_template(name):
        if name not in TEMPLATES:
            TEMPLATES[name] = template_env.get_template(name)
        return TEMPLATES[name]
    
    templates_available = True
except Exception as e:
    print(f"Warning: Jinja2 not available: {e}")
    templates_available = False

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
    version="0.8.0",
    description="Text-to-Speech API with advanced analytics, optimized model parsing, and enhanced UI"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global instance
current_tts_instance = None

# Directories
AUDIO_DIR = Path(tempfile.gettempdir()) / "stts_audio"
AUDIO_DIR.mkdir(exist_ok=True)
HISTORY_FILE = AUDIO_DIR / "history.json"

ANALYTICS_DIR = AUDIO_DIR / "analytics"
ANALYTICS_DIR.mkdir(exist_ok=True)
ANALYTICS_FILE = ANALYTICS_DIR / "reachability.json"
SYSTEM_STATS_FILE = ANALYTICS_DIR / "system_stats.json"
MODELS_CACHE_FILE = ANALYTICS_DIR / "models_cache.json"
MODELS_METADATA_FILE = ANALYTICS_DIR / "models_metadata.json"

# Analytics data
analytics_data = {
    "checks": [],
    "endpoints": {},
    "metrics": {
        "uptime": 99.5,
        "avgResponseTime": 245,
        "successRate": 98.2,
        "cpuLoad": 45,
        "memoryUsage": 62,
        "requestRate": 0,
        "errorRate": 1.8
    },
    "timeline": [],
    "lastUpdated": None,
    "systemStats": {}
}

MAX_CHECKS = 10000

request_counter = {
    "count": 0,
    "totalResponseTime": 0,
    "last_reset": time.time(),
    "errors": 0,
    "successes": 0
}

SERVER_START_TIME = time.time()
response_times_history = deque(maxlen=1000)
error_history = deque(maxlen=100)

# Models cache
MODELS_CACHE_FILE = ANALYTICS_DIR / "models_cache.json"
MODELS_CACHE = {}
MODELS_CACHE_TIME = 0

# Settings file
SETTINGS_FILE = AUDIO_DIR / "user_settings.json"


def load_user_settings():
    """Load user settings from file."""
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading settings: {e}")
    return {
        "put_accent": True,
        "put_yo": True,
        "auto_play": True,
        "num_threads": 6,
        "sample_rate": 48000,
        "device": "cpu"
    }


def save_user_settings(settings):
    """Save user settings to file."""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        print(f"Settings saved to {SETTINGS_FILE}")
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

# Performance benchmarking data
benchmark_data = {
    "model_load_times": {},
    "tts_generation_times": [],
    "memory_usage_history": [],
    "cpu_usage_history": [],
    "last_benchmark": None
}


# ========================================
# Helper Functions
# ========================================

def load_history():
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading history: {e}")
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving history: {e}")

def add_to_history(text, language, model_id, speaker, sample_rate, duration, file_path):
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
    if len(history) > 100:
        history = history[:100]
    
    save_history(history)
    return entry


def track_request(endpoint, success=True, response_time=0, status_code=200, details="OK", request_info: Dict = None):
    global analytics_data, request_counter
    
    check = {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "responseTime": int(response_time * 1000),
        "statusCode": status_code,
        "endpoint": endpoint,
        "details": details,
        "requestInfo": request_info or {}
    }

    analytics_data["checks"].append(check)

    if len(analytics_data["checks"]) > MAX_CHECKS:
        analytics_data["checks"] = analytics_data["checks"][-MAX_CHECKS:]
    
    if not success:
        analytics_data["timeline"].insert(0, {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "title": "Request Failed",
            "description": f"{endpoint} - {details}",
            "statusCode": status_code
        })
        if len(analytics_data["timeline"]) > 100:
            analytics_data["timeline"] = analytics_data["timeline"][:100]
    
    if endpoint not in analytics_data["endpoints"]:
        analytics_data["endpoints"][endpoint] = {
            "requests": 0,
            "totalResponseTime": 0,
            "success": 0,
            "errors": 0
        }
    
    analytics_data["endpoints"][endpoint]["requests"] += 1
    analytics_data["endpoints"][endpoint]["totalResponseTime"] += response_time
    if success:
        analytics_data["endpoints"][endpoint]["success"] += 1
    else:
        analytics_data["endpoints"][endpoint]["errors"] += 1
    
    request_counter["count"] += 1
    request_counter["totalResponseTime"] += response_time
    if success:
        request_counter["successes"] += 1
    else:
        request_counter["errors"] += 1
    
    if response_time > 0:
        response_times_history.append(response_time * 1000)
    
    if time.time() - request_counter["last_reset"] > 60:
        calculate_metrics()
        request_counter["count"] = 0
        request_counter["totalResponseTime"] = 0
        request_counter["errors"] = 0
        request_counter["successes"] = 0
        request_counter["last_reset"] = time.time()
    
    if len(analytics_data["checks"]) % 10 == 0:
        save_analytics()


def calculate_metrics():
    global analytics_data
    
    metrics = calculate_advanced_metrics()
    analytics_data["metrics"] = {
        "uptime": metrics.get("uptime", 99.5),
        "avgResponseTime": metrics.get("avgResponseTime", 245),
        "successRate": metrics.get("successRate", 98.2),
        "cpuLoad": metrics.get("cpuLoad", 45),
        "memoryUsage": metrics.get("memoryUsage", 62),
        "requestRate": metrics.get("requestRate", 0),
        "errorRate": metrics.get("errorRate", 1.8)
    }
    analytics_data["systemStats"] = metrics.get("systemStats", {})
    analytics_data["lastUpdated"] = datetime.now().isoformat()
    
    return metrics


def save_analytics():
    try:
        with open(ANALYTICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(analytics_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving analytics: {e}")


def load_analytics():
    global analytics_data
    try:
        if ANALYTICS_FILE.exists():
            with open(ANALYTICS_FILE, 'r', encoding='utf-8') as f:
                analytics_data = json.load(f)
    except Exception as e:
        print(f"Error loading analytics: {e}")


load_analytics()


def get_system_stats() -> Dict[str, Any]:
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net_io = psutil.net_io_counters()
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        server_uptime = time.time() - SERVER_START_TIME
        
        stats = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": round(cpu_percent, 2),
                "count": cpu_count,
                "freq_mhz": round(cpu_freq.current, 2) if cpu_freq else None,
                "load_1min": round(psutil.getloadavg()[0], 2) if hasattr(psutil, 'getloadavg') else None
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent": round(memory.percent, 2)
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": round(disk.percent, 2)
            },
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            },
            "uptime": {
                "system": str(timedelta(seconds=int(psutil.boot_time()))),
                "server": str(timedelta(seconds=int(server_uptime))),
                "server_seconds": int(server_uptime)
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
                "hostname": socket.gethostname()
            },
            "torch": {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        return stats
    except Exception as e:
        print(f"Error getting system stats: {e}")
        return {}


def calculate_advanced_metrics() -> Dict[str, Any]:
    checks = analytics_data["checks"]
    if not checks:
        return {}
    
    successful_checks = [c for c in checks if c.get("success", False)]
    total_checks = len(checks)
    
    success_rate = (len(successful_checks) / total_checks * 100) if total_checks > 0 else 100
    error_rate = 100 - success_rate
    
    response_times = [c.get("responseTime", 0) for c in successful_checks if c.get("responseTime", 0) > 0]
    
    if response_times:
        avg_response_time = mean(response_times)
        p50_response_time = sorted(response_times)[len(response_times)//2] if response_times else 0
        p95_response_time = sorted(response_times)[int(len(response_times)*0.95)] if len(response_times) > 20 else max(response_times)
        p99_response_time = sorted(response_times)[int(len(response_times)*0.99)] if len(response_times) > 100 else max(response_times)
        std_dev = stdev(response_times) if len(response_times) > 1 else 0
        min_response_time = min(response_times)
        max_response_time = max(response_times)
    else:
        avg_response_time = p50_response_time = p95_response_time = p99_response_time = std_dev = min_response_time = max_response_time = 0
    
    request_rate = request_counter["count"]
    current_error_rate = (request_counter["errors"] / max(1, request_counter["count"])) * 100
    system_stats = get_system_stats()
    
    metrics = {
        "uptime": round(success_rate, 2),
        "avgResponseTime": int(avg_response_time),
        "successRate": round(success_rate, 2),
        "errorRate": round(error_rate, 2),
        "requestRate": request_rate,
        "cpuLoad": system_stats.get("cpu", {}).get("percent", 0),
        "memoryUsage": system_stats.get("memory", {}).get("percent", 0),
        "responseTimeStats": {
            "avg_ms": round(avg_response_time, 2),
            "p50_ms": round(p50_response_time, 2),
            "p95_ms": round(p95_response_time, 2),
            "p99_ms": round(p99_response_time, 2),
            "min_ms": round(min_response_time, 2),
            "max_ms": round(max_response_time, 2),
            "stdDev_ms": round(std_dev, 2)
        },
        "requestStats": {
            "total": total_checks,
            "successful": len(successful_checks),
            "failed": total_checks - len(successful_checks),
            "rate_per_minute": request_rate,
            "errorRate_percent": round(current_error_rate, 2)
        },
        "systemStats": system_stats,
        "serverUptime": str(timedelta(seconds=int(time.time() - SERVER_START_TIME)))
    }
    
    return metrics


def save_system_stats():
    try:
        stats = get_system_stats()
        with open(SYSTEM_STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving system stats: {e}")


def get_health_status() -> Dict[str, Any]:
    metrics = calculate_advanced_metrics()
    
    health_score = 100
    issues = []
    
    avg_rt = metrics.get("responseTimeStats", {}).get("avg_ms", 0)
    if avg_rt > 1000:
        health_score -= 30
        issues.append({"level": "critical", "metric": "response_time", "value": f"{avg_rt}ms"})
    elif avg_rt > 500:
        health_score -= 15
        issues.append({"level": "warning", "metric": "response_time", "value": f"{avg_rt}ms"})
    
    error_rate = metrics.get("errorRate", 0)
    if error_rate > 10:
        health_score -= 40
        issues.append({"level": "critical", "metric": "error_rate", "value": f"{error_rate}%"})
    elif error_rate > 5:
        health_score -= 20
        issues.append({"level": "warning", "metric": "error_rate", "value": f"{error_rate}%"})
    
    cpu = metrics.get("cpuLoad", 0)
    if cpu > 90:
        health_score -= 25
        issues.append({"level": "critical", "metric": "cpu", "value": f"{cpu}%"})
    elif cpu > 70:
        health_score -= 10
        issues.append({"level": "warning", "metric": "cpu", "value": f"{cpu}%"})
    
    memory = metrics.get("memoryUsage", 0)
    if memory > 90:
        health_score -= 25
        issues.append({"level": "critical", "metric": "memory", "value": f"{memory}%"})
    elif memory > 80:
        health_score -= 10
        issues.append({"level": "warning", "metric": "memory", "value": f"{memory}%"})
    
    if health_score >= 90:
        status = "healthy"
    elif health_score >= 70:
        status = "degraded"
    else:
        status = "unhealthy"
    
    return {
        "status": status,
        "health_score": health_score,
        "timestamp": datetime.now().isoformat(),
        "issues": issues,
        "metrics": metrics,
        "version": "1.0.0"
    }


def render_template(template_name: str, context: Dict[str, Any] = None) -> str:
    try:
        template_path = TEMPLATES_DIR / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_name}")
        
        if templates_available:
            template = get_template(template_name)
            return template.render(context or {})
        else:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"Error rendering template {template_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Template error: {str(e)}")


# ========================================
# Optimized Model Parsing (Without Loading Models)
# ========================================

def get_models_metadata() -> Dict[str, Any]:
    """
    Get models metadata WITHOUT loading models into memory.
    Parses YAML config directly and filters out JIT models.
    """
    try:
        models_file = APP_DIR / "latest_silero_models.yml"
        
        if not models_file.exists():
            # Try to download
            if SILEROTTS_AVAILABLE:
                SileroTTS.download_models_config_static(str(models_file))
            else:
                return {}
        
        with open(models_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        metadata = {
            "languages": {},
            "total_models": 0,
            "jit_models_excluded": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        for lang, models in config.get('tts_models', {}).items():
            metadata["languages"][lang] = {}
            
            for model_id, model_info in models.items():
                # Check if this is a JIT model and exclude it
                latest = model_info.get('latest', {})
                package_url = latest.get('package', '')
                
                # Filter out JIT models
                if 'jit' in package_url.lower() or 'jit' in model_id.lower():
                    metadata["jit_models_excluded"] += 1
                    continue
                
                metadata["total_models"] += 1
                
                # Extract sample rates
                sample_rates = latest.get('sample_rate', [])
                if not isinstance(sample_rates, list):
                    sample_rates = [sample_rates]
                
                metadata["languages"][lang][model_id] = {
                    "package": package_url,
                    "sample_rates": sample_rates,
                    "has_speakers": model_id in ['multi_v2', 'v4_ru', 'v3_en', 'thorsten_v2'],
                    "estimated_size_mb": None  # Could be calculated from URL if needed
                }
        
        # Save metadata to file
        try:
            with open(MODELS_METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving models metadata: {e}")
        
        return metadata
    except Exception as e:
        print(f"Error getting models metadata: {e}")
        return {}


def get_models_cache(
    force_refresh: bool = False,
    language: Optional[str] = None,
    model_id: Optional[str] = None,
    load_speaker_info: bool = True
) -> Dict[str, Any]:
    """
    Get cached models list with optional filtering.
    
    Args:
        force_refresh: Bypass cache and reload from source
        language: Filter by language code
        model_id: Filter by model ID
        load_speaker_info: If False, don't load speaker info (faster, no model loading)
    
    Returns:
        Filtered cache data with metadata
    """
    global MODELS_CACHE, MODELS_CACHE_TIME
    
    # Use cache if not forced refresh and cache is fresh
    if not force_refresh and time.time() - MODELS_CACHE_TIME < 300 and MODELS_CACHE:
        result = MODELS_CACHE
        if language:
            result = {lang: data for lang, data in result.items() if lang == language}
        if model_id:
            result = {lang: {mid: info for mid, info in models.items() if model_id in mid or mid == model_id} 
                      for lang, models in result.items()}
        return result
    
    if not SILEROTTS_AVAILABLE:
        return {}
    
    try:
        cache_data = {}
        models = SileroTTS.get_available_models()
        
        # Filter by language
        if language:
            models = {language: models.get(language, [])}
        
        for lang, model_ids in models.items():
            cache_data[lang] = {}
            
            for mid in model_ids:
                # Filter by model_id
                if model_id and model_id not in mid and mid != model_id:
                    continue
                
                # Skip JIT models
                if 'jit' in mid.lower():
                    continue
                
                # Only load speaker info if requested
                if load_speaker_info:
                    try:
                        temp_tts = SileroTTS(
                            model_id=mid,
                            language=lang,
                            speaker=None,
                            sample_rate=48000,
                            device="cpu"
                        )
                        speaker_info = temp_tts.get_available_speakers()
                        speakers = speaker_info.get('speakers', []) if isinstance(speaker_info, dict) else []
                        sample_rates = speaker_info.get('sample_rates', []) if isinstance(speaker_info, dict) else []
                        
                        cache_data[lang][mid] = {
                            "speakers": speakers,
                            "sample_rates": sample_rates
                        }
                    except Exception as e:
                        print(f"Error caching speakers for {lang}/{mid}: {e}")
                        cache_data[lang][mid] = {"speakers": [], "sample_rates": []}
                        continue
                else:
                    # Don't load model, just get info from metadata
                    metadata = get_models_metadata()
                    if lang in metadata.get("languages", {}):
                        model_meta = metadata["languages"][lang].get(mid, {})
                        cache_data[lang][mid] = {
                            "speakers": [],  # Will be loaded on demand
                            "sample_rates": model_meta.get("sample_rates", [])
                        }
        
        MODELS_CACHE = cache_data
        MODELS_CACHE_TIME = time.time()
        
        # Always save to file
        try:
            with open(MODELS_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "data": cache_data,
                    "timestamp": datetime.now().isoformat(),
                    "filters": {"language": language, "model_id": model_id}
                }, f, indent=2, ensure_ascii=False)
            print(f"Models cache saved to {MODELS_CACHE_FILE}")
        except Exception as e:
            print(f"Error saving models cache: {e}")
        
        return cache_data
    except Exception as e:
        print(f"Error getting models cache: {e}")
        return {}


def load_models_cache_from_file():
    global MODELS_CACHE, MODELS_CACHE_TIME
    
    try:
        if MODELS_CACHE_FILE.exists():
            with open(MODELS_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                MODELS_CACHE = cache_data.get("data", {})
                MODELS_CACHE_TIME = time.time()
                print(f"Loaded models cache from file ({len(MODELS_CACHE)} languages)")
    except Exception as e:
        print(f"Error loading models cache from file: {e}")


load_models_cache_from_file()


# ========================================
# Benchmarking Functions
# ========================================

async def benchmark_model_loading():
    """Benchmark model loading times for all available models."""
    if not SILEROTTS_AVAILABLE:
        return {"error": "SileroTTS not available"}
    
    start_time = time.time()
    results = {
        "models": {},
        "total_time": 0,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        models = SileroTTS.get_available_models()
        languages = SileroTTS.get_available_languages()
        
        for lang in languages:
            results["models"][lang] = {}
            for model_id in models.get(lang, []):
                if 'jit' in model_id.lower():
                    continue
                
                model_start = time.time()
                try:
                    tts = SileroTTS(
                        model_id=model_id,
                        language=lang,
                        speaker=None,
                        sample_rate=24000,
                        device="cpu"
                    )
                    load_time = time.time() - model_start
                    
                    results["models"][lang][model_id] = {
                        "load_time_seconds": round(load_time, 3),
                        "speakers_count": len(tts.get_available_speakers()),
                        "status": "success"
                    }
                except Exception as e:
                    results["models"][lang][model_id] = {
                        "error": str(e),
                        "status": "failed"
                    }
        
        results["total_time"] = round(time.time() - start_time, 3)
        
        # Save benchmark results
        benchmark_data["model_load_times"] = results
        benchmark_data["last_benchmark"] = datetime.now().isoformat()
        
        return results
    except Exception as e:
        return {"error": str(e)}


async def benchmark_tts_generation(text: str = "Привет, это тестовая генерация речи.", model_id: str = "v4_ru", language: str = "ru", speaker: str = "kseniya_v2"):
    """Benchmark TTS generation performance."""
    if not SILEROTTS_AVAILABLE:
        return {"error": "SileroTTS not available"}
    
    results = []
    
    try:
        # Load model
        tts = SileroTTS(
            model_id=model_id,
            language=language,
            speaker=speaker,
            sample_rate=24000,
            device="cpu"
        )
        
        # Run multiple generations
        for i in range(3):
            start = time.time()
            
            output_file = str(AUDIO_DIR / f"benchmark_{i}.wav")
            tts.tts(text, output_file)
            
            generation_time = time.time() - start
            
            results.append({
                "iteration": i + 1,
                "generation_time_seconds": round(generation_time, 3),
                "text_length": len(text),
                "output_file": output_file
            })
        
        benchmark_data["tts_generation_times"] = results
        
        return {
            "benchmark": results,
            "avg_time": round(mean([r["generation_time_seconds"] for r in results]), 3),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


def get_realtime_system_monitoring():
    """Get real-time system monitoring data."""
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "network_io": psutil.net_io_counters()._asdict(),
        "torch_memory": {
            "allocated": torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0,
            "reserved": torch.cuda.memory_reserved(0) / 1024**2 if torch.cuda.is_available() else 0
        }
    }


# ========================================
# Pydantic Models
# ========================================

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


class OpenAITTSRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0


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
    if language == "en":
        return "en_0"
    return "kseniya_v2"


def get_or_create_tts_instance(
    model_id: str = None,
    language: str = "ru",
    speaker: str = None,
    sample_rate: int = 48000,
    device: str = "cpu"
) -> SileroTTS:
    global current_tts_instance
    
    if not model_id:
        model_id = get_model_id_for_language(language)
    
    if not speaker:
        speaker = get_default_speaker(language)
    
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


def generate_audio_file(tts: SileroTTS, text: str, file_id: str = None) -> str:
    if file_id is None:
        file_id = str(uuid.uuid4())
    
    output_path = AUDIO_DIR / f"tts_{file_id}.wav"
    result_file = tts.tts(str(text), str(output_path))
    return result_file


# ========================================
# API Endpoints
# ========================================

@app.get("/")
async def root(request: Request):
    """Root endpoint - enhanced dashboard with server exploration."""
    # Get comprehensive data for dashboard
    health = get_health_status()
    system_stats = get_system_stats()
    models_metadata = get_models_metadata()
    
    context = {
        "request": request,
        "health": health,
        "system_stats": system_stats,
        "models_metadata": models_metadata,
        "version": "1.0.0",
        "server_start": datetime.fromtimestamp(SERVER_START_TIME).isoformat()
    }
    
    html = render_template("main.html", context)
    return HTMLResponse(content=html)


@app.get("/ui")
async def get_ui(request: Request):
    """Modern web UI with cascading model selection."""
    html = render_template("index.html", {"request": request})
    return HTMLResponse(content=html)


@app.get("/data")
async def get_data(request: Request):
    """Analytics dashboard with visualizations."""
    calculate_metrics()
    html = render_template("data.html", {"request": request})
    return HTMLResponse(content=html)


@app.get("/api/analytics")
async def get_analytics():
    """Get analytics data in JSON format."""
    calculate_metrics()
    return analytics_data


@app.get("/api/health")
async def health_check(detailed: bool = Query(False)):
    """Health check endpoint."""
    return get_health_status() if detailed else {
        "status": get_health_status()["status"],
        "health_score": get_health_status()["health_score"],
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/api/system/info")
async def get_system_information():
    """Get detailed system information."""
    return get_system_stats()


@app.get("/api/models/metadata")
async def get_models_metadata_endpoint():
    """Get models metadata WITHOUT loading models (optimized)."""
    metadata = get_models_metadata()
    return {
        "success": True,
        "data": metadata,
        "cached": MODELS_CACHE_TIME > 0,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/models")
async def get_models(
    request: Request,
    language: Optional[str] = Query(None, description="Filter by language code"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    force_refresh: bool = Query(False, description="Bypass cache"),
    load_speakers: bool = Query(False, description="Load speaker info (slower)")
):
    """
    Get available models with filtering.
    
    Use load_speakers=false for fast metadata-only response.
    """
    start_time = time.time()
    
    if not SILEROTTS_AVAILABLE:
        track_request("/api/models", False, 0, 500, "SileroTTS not available")
        raise HTTPException(status_code=500, detail="SileroTTS library not available")
    
    try:
        # Always get fresh data if language changed
        if language and (not MODELS_CACHE or language not in MODELS_CACHE):
            force_refresh = True
        
        models_cache = get_models_cache(
            force_refresh=force_refresh,
            language=language,
            model_id=model_id,
            load_speaker_info=load_speakers
        )
        
        response_time = time.time() - start_time
        
        result = {
            "success": True,
            "data": models_cache,
            "filters": {
                "language": language,
                "model_id": model_id,
                "load_speakers": load_speakers
            },
            "cached": not force_refresh and MODELS_CACHE_TIME > 0,
            "cache_time": datetime.fromtimestamp(MODELS_CACHE_TIME).isoformat() if MODELS_CACHE_TIME else None,
            "response_time_ms": round(response_time * 1000, 2)
        }
        
        track_request("/api/models", True, response_time, 200, "OK")
        return result
        
    except Exception as e:
        track_request("/api/models", False, time.time() - start_time, 500, str(e))
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/models/{language}/{model_id}/speakers")
async def get_model_speakers(language: str, model_id: str):
    """Get speakers for a specific model (on-demand loading)."""
    if not SILEROTTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="SileroTTS not available")
    
    try:
        start = time.time()
        
        # Load model with error handling for numpy compatibility issues
        try:
            tts = SileroTTS(
                model_id=model_id,
                language=language,
                speaker=None,
                sample_rate=48000,
                device="cpu"
            )
        except (AttributeError, ImportError) as e:
            # Handle numpy/torch compatibility errors gracefully
            if "_signature_descriptor" in str(e) or "multiarray failed to import" in str(e):
                return {
                    "success": False,
                    "language": language,
                    "model_id": model_id,
                    "speakers": [],
                    "sample_rates": [],
                    "error": "Model loading failed due to numpy/torch compatibility. Please update your packages.",
                    "recommendation": "Try: pip install --upgrade numpy torch",
                    "load_time_ms": round((time.time() - start) * 1000, 2)
                }
            raise
        
        speaker_info = tts.get_available_speakers()
        
        # Handle different return formats from SileroTTS
        if isinstance(speaker_info, dict):
            speakers = speaker_info.get('speakers', [])
            sample_rates = speaker_info.get('sample_rates', [])
        elif isinstance(speaker_info, list):
            speakers = speaker_info
            sample_rates = tts.get_available_sample_rates() if hasattr(tts, 'get_available_sample_rates') else []
        else:
            # Fallback: try to get speakers from tts_model attribute
            if hasattr(tts, 'tts_model') and hasattr(tts.tts_model, 'speakers'):
                speakers = tts.tts_model.speakers
            else:
                speakers = []
            sample_rates = []
        
        # Ensure sample_rates are populated if empty
        if not sample_rates:
            try:
                sample_rates = tts.get_available_sample_rates()
            except:
                sample_rates = [48000, 24000]  # Default fallback
        
        response_time = time.time() - start
        
        return {
            "success": True,
            "language": language,
            "model_id": model_id,
            "speakers": speakers,
            "sample_rates": sample_rates,
            "load_time_ms": round(response_time * 1000, 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "language": language,
            "model_id": model_id,
            "speakers": [],
            "sample_rates": [],
            "error": str(e),
            "load_time_ms": round((time.time() - start) * 1000, 2)
        }


@app.get("/api/benchmark/loading")
async def benchmark_loading():
    """Benchmark model loading times."""
    result = await benchmark_model_loading()
    return result


@app.get("/api/benchmark/generation")
async def benchmark_generation(
    text: str = Query("Привет, это тестовая генерация речи.", description="Test text"),
    model_id: str = Query("v4_ru", description="Model to benchmark"),
    language: str = Query("ru", description="Language"),
    speaker: str = Query("kseniya_v2", description="Speaker")
):
    """Benchmark TTS generation performance."""
    result = await benchmark_tts_generation(text, model_id, language, speaker)
    return result


@app.get("/api/monitoring/realtime")
async def get_realtime_monitoring():
    """Get real-time system monitoring data."""
    try:
        stats = get_realtime_system_monitoring()
        
        # Ensure server uptime is always included
        stats["server_uptime"] = {
            "seconds": int(time.time() - SERVER_START_TIME),
            "formatted": str(timedelta(seconds=int(time.time() - SERVER_START_TIME)))
        }
        
        return stats
    except Exception as e:
        return {
            "error": str(e),
            "server_uptime": {
                "seconds": int(time.time() - SERVER_START_TIME),
                "formatted": str(timedelta(seconds=int(time.time() - SERVER_START_TIME)))
            }
        }


@app.get("/api/metrics")
async def get_metrics():
    """Prometheus-style metrics."""
    metrics = calculate_advanced_metrics()
    
    prometheus_format = f"""# HELP silerotts_uptime Server uptime percentage
# TYPE silerotts_uptime gauge
silerotts_uptime {metrics['uptime']}
# HELP silerotts_response_time_avg Average response time in milliseconds
# TYPE silerotts_response_time_avg gauge
silerotts_response_time_avg {metrics['responseTimeStats']['avg_ms']}
# HELP silerotts_response_time_p95 95th percentile response time
# TYPE silerotts_response_time_p95 gauge
silerotts_response_time_p95 {metrics['responseTimeStats']['p95_ms']}
# HELP silerotts_success_rate Success rate percentage
# TYPE silerotts_success_rate gauge
silerotts_success_rate {metrics['successRate']}
# HELP silerotts_error_rate Error rate percentage
# TYPE silerotts_error_rate gauge
silerotts_error_rate {metrics['errorRate']}
# HELP silerotts_requests_total Total number of requests
# TYPE silerotts_requests_total counter
silerotts_requests_total {metrics['requestStats']['total']}
# HELP silerotts_cpu_usage CPU usage percentage
# TYPE silerotts_cpu_usage gauge
silerotts_cpu_usage {metrics['cpuLoad']}
# HELP silerotts_memory_usage Memory usage percentage
# TYPE silerotts_memory_usage gauge
silerotts_memory_usage {metrics['memoryUsage']}
"""
    
    return HTMLResponse(content=prometheus_format, media_type="text/plain")


@app.get("/api/performance")
async def get_performance_stats():
    """Get detailed performance statistics."""
    metrics = calculate_advanced_metrics()
    
    return {
        "response_times": metrics.get("responseTimeStats", {}),
        "request_stats": metrics.get("requestStats", {}),
        "benchmark": benchmark_data,
        "server_uptime": metrics.get("serverUptime"),
        "last_updated": datetime.now().isoformat()
    }


@app.get("/api/history")
async def get_request_history(
    limit: int = Query(100, description="Number of recent requests"),
    endpoint: Optional[str] = Query(None, description="Filter by endpoint")
):
    """Get recent request history."""
    calculate_metrics()
    
    checks = analytics_data["checks"]
    
    if endpoint:
        checks = [c for c in checks if c.get("endpoint") == endpoint]
    
    checks = checks[-limit:]
    
    return {
        "requests": checks,
        "total": len(checks),
        "endpoint_filter": endpoint,
        "timestamp": datetime.now().isoformat()
    }


# ========================================
# OpenAI-Compatible Endpoints
# ========================================

@app.get("/v1/models")
async def get_openai_models(language: Optional[str] = Query(None)):
    """OpenAI-compatible models list endpoint."""
    if not SILEROTTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="SileroTTS library not available")
    
    try:
        models_cache = get_models_cache(language=language, load_speaker_info=True)
        
        openai_models = []
        
        for lang, model_data in models_cache.items():
            for model_id, info in model_data.items():
                speaker_list = info.get("speakers", [])
                
                for speaker in speaker_list:
                    openai_models.append({
                        "id": f"{model_id}_{speaker}",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "silerotts",
                        "metadata": {
                            "language": lang,
                            "model_id": model_id,
                            "speaker": speaker,
                            "type": "tts",
                            "sample_rates": info.get("sample_rates", [])
                        }
                    })
        
        return {
            "object": "list",
            "data": openai_models,
            "cached": True,
            "cache_time": datetime.fromtimestamp(MODELS_CACHE_TIME).isoformat() if MODELS_CACHE_TIME else None,
            "total": len(openai_models),
            "filters": {"language": language}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")


# ========================================
# Main TTS Endpoint
# ========================================

@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech from text."""
    start_time = time.time()
    
    if not SILEROTTS_AVAILABLE:
        track_request("/api/tts", False, 0, 500, "SileroTTS not available")
        raise HTTPException(status_code=500, detail="SileroTTS library not available")
    
    try:
        tts = get_or_create_tts_instance(
            model_id=request.model_id,
            language=request.language,
            speaker=request.speaker,
            sample_rate=request.sample_rate,
            device=request.device
        )
        
        file_id = str(uuid.uuid4())
        output_path = generate_audio_file(tts, request.text, file_id)
        
        # Convert absolute path to relative URL path
        # output_path is like "C:\Users\...\Temp\stts_audio\tts_abc.wav"
        # We need to extract just the filename for the URL
        filename = Path(output_path).name
        audio_url = f"/api/audio/{filename}"
        
        duration = time.time() - start_time
        
        # Add to history
        add_to_history(
            text=request.text,
            language=tts.language,
            model_id=tts.model_id,
            speaker=tts.speaker,
            sample_rate=tts.sample_rate,
            duration=duration,
            file_path=output_path
        )
        
        track_request("/api/tts", True, duration, 200, f"Generated {len(request.text)} chars")
        
        return {
            "success": True,
            "file_path": output_path,
            "audio_url": audio_url,  # URL for browser to load
            "filename": filename,
            "duration_seconds": round(duration, 3),
            "model": tts.model_id,
            "language": tts.language,
            "speaker": tts.speaker,
            "sample_rate": tts.sample_rate
        }
    except Exception as e:
        track_request("/api/tts", False, time.time() - start_time, 500, str(e))
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve generated audio files."""
    audio_file = AUDIO_DIR / filename
    
    if not audio_file.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=str(audio_file),
        filename=filename,
        media_type="audio/wav"
    )


@app.get("/api/settings")
async def get_settings():
    """Get current user settings."""
    return {
        "success": True,
        "settings": load_user_settings()
    }


@app.post("/api/settings")
async def save_settings(settings: Dict[str, Any]):
    """Save user settings."""
    success = save_user_settings(settings)
    return {
        "success": success,
        "settings": settings
    }


@app.on_event("startup")
async def startup_event():
    """Run on server startup."""
    print("=" * 60)
    print("SileroTTS API Server v1.0.0")
    print("=" * 60)
    print(f"Server starting at: {datetime.now().isoformat()}")
    print(f"Audio directory: {AUDIO_DIR}")
    print(f"Models cache: {MODELS_CACHE_FILE}")
    print(f"SileroTTS available: {SILEROTTS_AVAILABLE}")

    if SILEROTTS_AVAILABLE:
        models = SileroTTS.get_available_models()
        languages = SileroTTS.get_available_languages()
        print(f"Languages: {languages}")
        print(f"Total models: {sum(len(m) for m in models.values())}")

    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
