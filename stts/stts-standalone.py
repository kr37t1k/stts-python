#!/usr/bin/env python3
import argparse
import os
import re
import sys
import timeit
import wave
import sounddevice as sd, soundfile as sf
from datetime import datetime, timedelta

# --- Configurable Parameters (via code or CLI) ---
# These can be overridden via command-line arguments
MODEL_ID: str = 'v4_ru'  # e.g., 'v3_en', 'v4_ru', see models.yml
LANGUAGE: str = 'ru'      # e.g., 'en', 'ru', 'de', see models.yml
SPEAKER: str = 'kseniya'       # e.g., 'lj', 'xenia', 'thorsten', depends on model/lang
SAMPLE_RATE: int = 24000  # Hz - 48000, 24000, 16000, 8000 (lower is less resource-intensive)
TORCH_DEVICE: str = 'cpu' # Use 'cpu' for mobile stability, 'cuda' if available
TORCH_NUM_THREADS: int = 4 # For CPU - tune based on your device (4-6 often good)
PUT_ACCENT: bool = True   # For some languages/models
PUT_YO: bool = True       # For Russian models
LINE_LENGTH_LIMIT: int = 1000 # Adjust based on speaker/model (check models.yml or find empirically)
WAVE_FILE_SIZE_LIMIT: int = 512 * 1024 * 1024  # 512 MiB - Max file size

# --- Global Constants ---
WAVE_CHANNELS: int = 1  # Mono
WAVE_HEADER_SIZE: int = 44  # Bytes
WAVE_SAMPLE_WIDTH: int = int(16 / 8)  # 16 bits == 2 bytes

# --- Import with Error Handling ---
try:
    import torch
    import omegaconf
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install torch num2t4ru (for Russian) omegaconf")
    sys.exit(1)

def main():
    global MODEL_ID, LANGUAGE, SPEAKER, SAMPLE_RATE, TORCH_DEVICE, TORCH_NUM_THREADS, PUT_ACCENT, PUT_YO, LINE_LENGTH_LIMIT, WAVE_FILE_SIZE_LIMIT
    # Parse command-line arguments to override defaults
    parser = argparse.ArgumentParser(description="Enhanced Silero TTS Standalone Script")
    parser.add_argument("input", help="Text to synthesize or path to the input text file")
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help=f"Model ID (default: {MODEL_ID})")
    parser.add_argument("--language", type=str, default=LANGUAGE, help=f"Language (default: {LANGUAGE})")
    parser.add_argument("--speaker", type=str, default=SPEAKER, help=f"Speaker (default: {SPEAKER})")
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE, help=f"Sample Rate (default: {SAMPLE_RATE})")
    parser.add_argument("--device", type=str, default=TORCH_DEVICE, help=f"Torch Device (default: {TORCH_DEVICE})")
    parser.add_argument("--threads", type=int, default=TORCH_NUM_THREADS, help=f"Torch Num Threads (default: {TORCH_NUM_THREADS})")
    parser.add_argument("--put_accent", action='store_true', default=PUT_ACCENT, help="Put accent (default: True if enabled)")
    parser.add_argument("--no_put_accent", dest='put_accent', action='store_false', help="Do not put accent")
    parser.add_argument("--put_yo", action='store_true', default=PUT_YO, help="Put yo (default: True if enabled)")
    parser.add_argument("--no_put_yo", dest='put_yo', action='store_false', help="Do not put yo")
    parser.add_argument("--line_length_limit", type=int, default=LINE_LENGTH_LIMIT, help=f"Max line length (default: {LINE_LENGTH_LIMIT})")
    parser.add_argument("--output_prefix", type=str, default="temp", help="Prefix for output files (default: input filename)")

    args = parser.parse_args()

    # Update global variables based on arguments
    MODEL_ID = args.model_id
    LANGUAGE = args.language
    SPEAKER = args.speaker
    SAMPLE_RATE = args.sample_rate
    TORCH_DEVICE = args.device
    TORCH_NUM_THREADS = args.threads
    PUT_ACCENT = args.put_accent
    PUT_YO = args.put_yo
    LINE_LENGTH_LIMIT = args.line_length_limit
    output_prefix = args.output_prefix if args.output_prefix else os.path.splitext(os.path.basename(args.input))[0]

    print(f"Starting TTS with args: Model={MODEL_ID}, Lang={LANGUAGE}, Speaker={SPEAKER}, SR={SAMPLE_RATE}, Device={TORCH_DEVICE}, Threads={TORCH_NUM_THREADS}")
    print(f"Accents/Yo: {PUT_ACCENT}/{PUT_YO}, Line Limit: {LINE_LENGTH_LIMIT}, Output Prefix: {output_prefix}")

    # --- Load Input File ---
    try:
        if args.input.endswith('.txt'):
            origin_lines = load_file(args.input)
        else:
            origin_lines = [args.input]
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading text: {e}")
        sys.exit(1)

    # --- Preprocess Text ---
    try:
        preprocessed_lines, preprocessed_text_len = preprocess_text(origin_lines, LINE_LENGTH_LIMIT)
        # Optionally save preprocessed text for debugging
        # write_lines(args.input_file + '_preprocessed.txt', preprocessed_lines)
    except Exception as e:
        print(f"Error during text preprocessing: {e}")
        sys.exit(1)

    # --- Initialize Model (Enhanced for Stability) ---
    print("Initializing Silero TTS model...")
    try:
        tts_model = init_model(TORCH_DEVICE, TORCH_NUM_THREADS)
        print(f"Available speakers for model: {tts_model.speakers}")
        if SPEAKER not in tts_model.speakers:
            print(f"Warning: Speaker '{SPEAKER}' not found in available speakers: {tts_model.speakers}. Using first available speaker.")
            SPEAKER = tts_model.speakers[0] # Fallback
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)

    # --- Process TTS ---
    print("Starting TTS synthesis...")
    try:
        process_tts(tts_model, preprocessed_lines, output_prefix, WAVE_FILE_SIZE_LIMIT, preprocessed_text_len)
        print("TTS synthesis completed successfully.")
    except KeyboardInterrupt:
        print("\nTTS synthesis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error during TTS synthesis: {e}")
        # Consider adding more specific error handling here if needed
        sys.exit(1)


def load_file(filename: str) -> list:
    print(f"Loading file: {filename}")
    try:
        with open(filename, 'r', encoding='utf-8') as f: # Explicitly set encoding
            lines = f.readlines()
        return lines
    except UnicodeDecodeError:
        print(f"UTF-8 decode failed for {filename}, trying latin-1...")
        with open(filename, 'r', encoding='latin-1') as f:
            lines = f.readlines()
        return lines
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        raise # Re-raise to be caught by main()


def spell_digits(line: str, lang: str = LANGUAGE) -> str:
    """Enhanced digit spelling based on language."""
    if lang == 'ru':
        try:
            digits = re.findall(r'\d+', line)
            # Sort digits from largest to smallest
            digits = sorted(digits, key=len, reverse=True)
            for digit in digits:
                 # Limit to 12 digits for num2t4ru
                 line = line.replace(digit, num2text(int(digit[:12])), 1) # Replace one at a time to avoid overlap issues
        except (ValueError, IndexError):
            print(f"Warning: Could not convert digit '{digit[:12]}' to text, leaving as-is.")
            # If num2t4ru fails, leave the digit as is
            pass
    else:
        # For other languages, basic handling (could be expanded)
        # Example: Replace percentages, decimals, etc., as needed per language
        line = re.sub(r'(\d+)\.(\d+)', r'\1 point \2', line) # Basic decimal handling for non-Russian
    return line


def preprocess_text(lines: list, length_limit: int) -> (list, int):
    print(f"Preprocessing text with line length limit={length_limit}")
    if length_limit > 3:
        length_limit = length_limit - 2  # Keep a room for trailing char and '\n' char
    else:
        print(f"ERROR: line length limit must be >= 3, got {length_limit}")
        exit(1)

    preprocessed_text_len = 0
    preprocessed_lines = []
    for line_num, line in enumerate(lines):
        line = line.strip()  # Remove leading/trailing spaces
        if not line: # Handles empty lines or just '\n'
            continue

        # --- Preprocessing based on language/model specifics ---
        # Replace chars not supported by many models
        line = line.replace("…", "...")  # Common replacement
        line = line.replace("*", " star ") # Example replacement
        line = re.sub(r'(\d+)[,](\d+)', r'\1 \2', line) # Handle commas in numbers if needed, context-dependent
        line = line.replace("%", " percent ") # Example replacement
        # Add more language-specific replacements here if needed

        # Spell digits (language-aware)
        line = spell_digits(line, LANGUAGE)

        # --- Splitting Logic ---
        while len(line) > 0:
            if len(line) <= length_limit:
                line = line + "\n"
                preprocessed_lines.append(line)
                preprocessed_text_len += len(line)
                break

            # Find position to split line between sentences/punctuation
            split_pos = 0
            for char in ['.', '!', '?', ';', ':', ',']: # Order matters
                pos = line.rfind(char, 0, length_limit)
                if pos > split_pos:
                    split_pos = pos
            # If punctuation found, split after it
            if split_pos > 0:
                 part = line[:split_pos + 1] + "\n"
                 preprocessed_lines.append(part)
                 preprocessed_text_len += len(part)
                 line = line[split_pos + 1:].lstrip() # Remove leading spaces after split
                 continue

            # If no punctuation found, try splitting on space
            split_pos = line.rfind(' ', 0, length_limit)
            if split_pos > 0:
                 part = line[:split_pos] + "\n"
                 preprocessed_lines.append(part)
                 preprocessed_text_len += len(part)
                 line = line[split_pos + 1:].lstrip() # Remove leading spaces after split
                 continue

            # If no space found, force split at limit
            print(f"Warning: Forcing split at line {line_num+1} at char {length_limit}. This might cause unnatural breaks.")
            part = line[:length_limit] + "\n"
            preprocessed_lines.append(part)
            preprocessed_text_len += len(part)
            line = line[length_limit:].lstrip() # Remove leading spaces after forced split

    return preprocessed_lines, preprocessed_text_len


def init_model(device: str, threads_count: int) -> torch.nn.Module:
    # --- Disable JIT profiling warnings (addresses GitHub issue #183) ---
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)

    # --- Device Setup ---
    if not torch.cuda.is_available():
        device = 'cpu'
    torch_device = torch.device(device)
    torch.set_num_threads(threads_count)
    print(f"Using device: {torch_device}, threads: {threads_count}")

    # --- Load Model ---
    print(f"Loading model '{MODEL_ID}' for language '{LANGUAGE}'...")
    # Use a temporary directory for cache to avoid permission issues on mobile
    cache_dir = os.path.expanduser("~/.cache/silero_tts") # Use user's home cache
    os.makedirs(cache_dir, exist_ok=True) # Ensure cache dir exists

    try:
        tts_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=LANGUAGE,
            speaker=MODEL_ID,
            cache_dir=cache_dir # Explicitly set cache dir
        )
    except Exception as e:
        print(f"Error loading model from hub: {e}")
        # Optionally, you could try loading a local model file if available
        # tts_model = torch.jit.load('path/to/local/model.pt')
        raise # Re-raise to be handled by main()

    # --- Move Model to Device ---
    print(f"Moving model to device: {torch_device}...")
    tts_model.to(torch_device)

    print("Model loaded and ready.")
    return tts_model


def init_wave_file(name: str, channels: int, sample_width: int, rate: int):
    print(f'Initializing wave file: {name}')
    wf = wave.open(name, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(rate)
    return wf


class Stats:
    def __init__(self, preprocessed_text_len: int):
        self.start_time = int(datetime.now().timestamp())
        self.preprocessed_text_len = preprocessed_text_len
        self.processed_text_len = 0
        self.done_percent = 0.0
        self.warmup_seconds = 0
        self.run_time = "0:00:00"
        self.run_time_est = "0:00:00"
        self.wave_data_current = 0
        self.wave_data_total = 0
        self.wave_mib = 0
        self.wave_mib_est = 0
        self.tts_time = "0:00:00"
        self.tts_time_est = "0:00:00"
        self.tts_time_current = "0:00:00"
        self.line_number = 0
        self.first_tts_start_time = None # Track for warmup

    def update(self, line: str, audio_chunk_size: int):
        self.line_number += 1
        self.wave_data_total += audio_chunk_size
        self.wave_data_current += audio_chunk_size
        self.processed_text_len += len(line)

        # Percentage calculation
        self.done_percent = round(self.processed_text_len * 100 / self.preprocessed_text_len, 1)

        # Wave size estimation
        self.wave_mib = int((self.wave_data_total / 1024 / 1024))
        self.wave_mib_est = int(
            (self.wave_data_total / 1024 / 1024 * self.preprocessed_text_len / self.processed_text_len))

        # Warmup estimation (using first TTS call)
        if self.first_tts_start_time is None:
            self.first_tts_start_time = int(datetime.now().timestamp())

        # Run time estimation
        current_time = int(datetime.now().timestamp())
        if self.first_tts_start_time:
             run_time_s = current_time - self.first_tts_start_time
             run_time_est_s = int(run_time_s * self.preprocessed_text_len / max(self.processed_text_len, 1)) # Avoid div by 0
             self.run_time = str(timedelta(seconds=run_time_s))
             self.run_time_est = str(timedelta(seconds=run_time_est_s))
        else:
             self.run_time = "0:00:00"
             self.run_time_est = "0:00:00"


        # TTS time estimation (based on audio length)
        tts_time_s = int((self.wave_data_total / WAVE_CHANNELS / WAVE_SAMPLE_WIDTH / SAMPLE_RATE))
        tts_time_est_s = int((tts_time_s * self.preprocessed_text_len / max(self.processed_text_len, 1))) # Avoid div by 0
        self.tts_time = str(timedelta(seconds=tts_time_s))
        self.tts_time_est = str(timedelta(seconds=tts_time_est_s))
        tts_time_current_s = int((self.wave_data_current / WAVE_CHANNELS / WAVE_SAMPLE_WIDTH / SAMPLE_RATE))
        self.tts_time_current = str(timedelta(seconds=tts_time_current_s))

    def next_file(self):
        self.wave_data_current = 0


def process_tts(tts_model: torch.nn.Module, lines: list, output_prefix: str, wave_data_limit: int, preprocessed_text_len: int):
    s = Stats(preprocessed_text_len)
    audio_size = WAVE_HEADER_SIZE
    wave_file_number = 0
    current_line_idx = 0

    # Generate initial output filename
    output_filename = f'{output_prefix}_{SPEAKER}_{wave_file_number}.wav'
    wf = init_wave_file(output_filename, WAVE_CHANNELS, WAVE_SAMPLE_WIDTH, SAMPLE_RATE)
    print(f"Writing to: {output_filename}")

    for line in lines:
        if not line.strip(): # Skip empty lines after stripping
            continue

        # Progress and status print
        if current_line_idx % 10 == 0: # Print every 10 lines or adjust frequency
             print(
                 f'Line {current_line_idx+1}/{len(lines)} | '
                 f'Done: {s.done_percent:.1f}% | '
                 f'Time: {s.run_time}/{s.run_time_est} | '
                 f'TTS: {s.tts_time_current}/{s.tts_time_est} | '
                 f'File: {s.wave_mib}/{s.wave_mib_est} MiB'
             )
        # Optional: print current line being processed (can be noisy)
        # print(f"Processing: {line.strip()}")

        try:
            # --- Apply TTS ---
            audio = tts_model.apply_tts(
                text=line.strip(), # Ensure no leading/trailing newlines passed to TTS
                speaker=SPEAKER,
                sample_rate=SAMPLE_RATE,
                put_accent=PUT_ACCENT,
                put_yo=PUT_YO
            )

            # --- Handle Audio Output ---
            if audio is not None and len(audio) > 0:
                audio_int16 = (audio * 32767).numpy().astype('int16')
                audio_bytes = audio_int16.tobytes()
                chunk_size_bytes = len(audio_bytes)

                # Check if adding this chunk exceeds the file size limit
                if audio_size + chunk_size_bytes > wave_data_limit:
                    print(f"File size limit ({wave_data_limit} bytes) reached. Starting new file...")
                    wf.close()
                    s.next_file()
                    wave_file_number += 1
                    audio_size = WAVE_HEADER_SIZE # Reset size for new file
                    output_filename = f'{output_prefix}_{SPEAKER}_{wave_file_number}.wav'
                    wf = init_wave_file(output_filename, WAVE_CHANNELS, WAVE_SAMPLE_WIDTH, SAMPLE_RATE)
                    print(f"Writing to: {output_filename}")

                # Write the audio chunk
                wf.writeframes(audio_bytes)
                audio_size += chunk_size_bytes
                s.update(line, chunk_size_bytes) # Update stats with actual bytes written
            else:
                print(f"Warning: TTS returned no audio for line: {line[:50]}...") # Log short snippet
                s.update(line, 0) # Still update stats for progress, even if no audio

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                 print(f"RuntimeError (likely OOM): {e}")
                 print(f"Consider reducing sample_rate, using 'cpu' device, or simplifying the input line: {line[:50]}...")
                 # Optionally, try to continue with next line or exit gracefully
                 s.update(line, 0) # Update stats and continue
            else:
                 print(f"RuntimeError during TTS for line: {line[:50]}... Error: {e}")
                 # Decide: continue, exit, or try fallback (e.g., skip line)
                 s.update(line, 0) # For now, update and continue
        except ValueError as e:
             print(f"ValueError during TTS for line: {line[:50]}... Error: {e}")
             # Handle potential model-specific errors (e.g., unsupported characters after preprocessing)
             s.update(line, 0) # Update stats and continue
        except Exception as e:
             print(f"Unexpected error during TTS for line: {line[:50]}... Error: {e}")
             s.update(line, 0) # Update stats and continue

        current_line_idx += 1

    # Close the final wave file
    wf.close()
    print(f"Final wave file closed: {output_filename}")
    sf_file, sf_samplerate = sf.read(output_filename)
    sd.play(data=sf_file, samplerate=SAMPLE_RATE, blocking=True)
    sd.wait()
    print(f"Wave export file playing: {output_filename}...")


if __name__ == "__main__":
    main()