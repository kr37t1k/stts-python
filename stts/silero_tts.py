import os
import re
import timeit
import torch
import wave
import yaml
import requests
import sounddevice as sd
import soundfile as sf
from loguru import logger
from datetime import datetime, timedelta
from number2text.number2text import NumberToText
from stts.lang_data import is_cyrillic, is_latin, lang_data
from stts.transliterate import reverse_transliterate, transliterate

class SileroTTS:
    def __init__(self, model_id: str, language: str, speaker: str = None, sample_rate: int = 48000, device: str = 'cpu',
                 put_accent=True, put_yo=True, num_threads=6):
        """
        Initialize the SileroTTS model with proper validation and error handling.
        
        Args:
            model_id (str): ID of the model to use
            language (str): Language code for the model
            speaker (str, optional): Speaker name. If None, first available speaker is used
            sample_rate (int): Audio sample rate (default: 48000)
            device (str): Device to use ('cpu', 'cuda', 'auto') (default: 'cpu')
            put_accent (bool): Whether to put accents (default: True)
            put_yo (bool): Whether to put 'yo' (default: True)
            num_threads (int): Number of threads for torch (default: 6)
        """
        if not model_id or not isinstance(model_id, str):
            raise ValueError("model_id must be a non-empty string")
        if not language or not isinstance(language, str):
            raise ValueError("language must be a non-empty string")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if device not in ['cpu', 'cuda', 'auto']:
            raise ValueError("device must be one of 'cpu', 'cuda', or 'auto'")
        if not isinstance(num_threads, int) or num_threads <= 0:
            raise ValueError("num_threads must be a positive integer")
            
        self.model_id = model_id
        self.language = language
        self.sample_rate = sample_rate
        self.device = device
        self.put_accent = put_accent
        self.put_yo = put_yo
        self.num_threads = num_threads

        self.models_config = self.load_models_config()
        self.tts_model = self.init_model()

        if speaker is None:
            if hasattr(self.tts_model, 'speakers') and self.tts_model.speakers:
                self.speaker = self.tts_model.speakers[0]
            else:
                # For models that don't have a speakers attribute, set a default speaker
                # This allows multi_v2 models to work without requiring explicit speaker specification
                if self.language == 'ru':
                    self.speaker = 'kseniya_v2'  # Use a common Russian v2 speaker
                elif self.language == 'en':
                    self.speaker = 'en_0'  # Use a common English speaker
                else:
                    self.speaker = 'random'  # fallback
                logger.info(f"Model doesn't expose speakers list, using default speaker: {self.speaker}")
        else:
            self.speaker = speaker

        self.validate_model()

        self.converter = NumberToText(self.language)
        self.wave_channels = 1  # Mono
        self.wave_header_size = 44  # Bytes
        self.wave_sample_width = int(16 / 8)  # 16 bits == 2 bytes
        self.LINE_LENGTH_LIMIT = 1000  # Default line length limit
        self.WAVE_FILE_SIZE_LIMIT = 512 * 1024 * 1024  # 512 MiB - Max file size

    def load_models_config(self):
        models_file = os.path.join(os.path.dirname(__file__), 'latest_silero_models.yml')

        if not os.path.exists(models_file):
            logger.warning(f"Models config file not found: {models_file}. Downloading...")
            self.download_models_config(models_file)

        with open(models_file, 'r', encoding='utf-8') as f:
            models_config = yaml.safe_load(f)
        logger.success(f"Models config loaded from: {models_file}")
        return models_config

    def download_models_config(self, models_file=None):
        """
        Download the models configuration file with retry and error handling.
        """
        url = "https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml"
        
        if models_file is None:
            models_file = os.path.join(os.path.dirname(__file__), 'models.yml')

        try:
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        with open(models_file, 'w', encoding='utf-8') as f:
                            f.write(response.text)
                        logger.success(f"Models config file downloaded: {models_file}")
                        return
                    else:
                        logger.warning(f"Failed to download models config file. Status code: {response.status_code}. Attempt {attempt + 1}/{max_retries}")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Network error during download: {str(e)}. Attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
            
            logger.error(f"Failed to download models config file after {max_retries} attempts")
            raise Exception(f"Failed to download models config file after {max_retries} attempts")
            
        except Exception as e:
            logger.error(f"Error downloading models config: {str(e)}")
            raise

    def get_available_speakers(self):
        if hasattr(self.tts_model, 'speakers'):
            return self.tts_model.speakers
        else:
            # Return default speakers list if the model doesn't have a speakers attribute
            if self.language == 'ru':
                return ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']
            elif self.language == 'en':
                return ['en_0', 'en_1', 'en_2', 'en_3', 'en_4', 'en_5', 'en_6', 'en_7', 'en_8', 'en_9', 'en_10', 'random']
            else:
                return ['random']  # fallback for other languages
    
    def get_available_sample_rates(self):
        model_config = self.models_config['tts_models'][self.language][self.model_id]['latest']
        sample_rates = model_config.get('sample_rate', [])

        if not isinstance(sample_rates, list):
            sample_rates = [sample_rates]

        return sample_rates

    def validate_model(self):
        model_config = self.models_config['tts_models'][self.language][self.model_id]['latest']

        if self.sample_rate not in model_config['sample_rate']:
            logger.error(f"Sample rate {self.sample_rate} is not supported for model '{self.model_id}'. Supported sample rates: {model_config['sample_rate']}")
            raise ValueError(f"Sample rate {self.sample_rate} is not supported for model '{self.model_id}'. Supported sample rates: {model_config['sample_rate']}")

        # Check if the model has a speakers attribute before trying to validate the speaker
        if hasattr(self.tts_model, 'speakers'):
            if self.speaker and self.speaker not in self.tts_model.speakers:
                logger.error(f"Speaker '{self.speaker}' is not supported for model '{self.model_id}'. Supported speakers: {self.tts_model.speakers}")
                raise ValueError(f"Speaker '{self.speaker}' is not supported for model '{self.model_id}'. Supported speakers: {self.tts_model.speakers}")
        else:
            # For models without a speakers attribute, we can't validate the speaker
            # But we'll allow it to proceed with the assumption that the model supports the requested speaker
            logger.info(f"Model '{self.model_id}' doesn't expose speakers list, proceeding with speaker '{self.speaker}'")

    
    def change_language(self, language):
        if language not in self.get_available_languages():
            logger.error(f"Language '{language}' is not supported.")
            logger.info(f"Available languages: {', '.join(self.get_available_languages())}")
            return

        self.language = language
        self.model_id = self.get_latest_model(language)
        available_sample_rates = self.get_available_sample_rates()
        self.sample_rate = max(available_sample_rates)
        self.tts_model,_= self.init_model()
        
        # Set speaker based on whether model has speakers attribute
        if hasattr(self.tts_model, 'speakers') and self.tts_model.speakers:
            self.speaker = self.tts_model.speakers[0]
        else:
            # For models without speakers attribute, use a default
            if self.language == 'ru':
                self.speaker = 'kseniya_v2'
            elif self.language == 'en':
                self.speaker = 'en_0'
            else:
                self.speaker = 'random'
            logger.info(f"Model doesn't expose speakers list, using default speaker: {self.speaker}")
        
        self.validate_model()

        logger.success(f"Language changed to: {language}. Using the latest model: {self.model_id}")
        logger.info(f"Available speakers for the new model: {', '.join(self.get_available_speakers())}")
        logger.info(f"Sample rate set to the highest available: {self.sample_rate}")

    def change_model(self, model_id):
        if model_id not in self.get_available_models()[self.language]:
            logger.error(f"Model '{model_id}' is not available for language '{self.language}'.")
            logger.info(f"Available models for {self.language}: {', '.join(self.get_available_models()[self.language])}")
            return

        self.model_id = model_id
        available_sample_rates = self.get_available_sample_rates()
        self.sample_rate = max(available_sample_rates)
        self.tts_model,_ = self.init_model()
        
        # Set speaker based on whether model has speakers attribute
        if hasattr(self.tts_model, 'speakers') and self.tts_model.speakers:
            self.speaker = self.tts_model.speakers[0]
        else:
            # For models without speakers attribute, use a default
            if self.language == 'ru':
                self.speaker = 'kseniya_v2'
            elif self.language == 'en':
                self.speaker = 'en_0'
            else:
                self.speaker = 'random'
            logger.info(f"Model doesn't expose speakers list, using default speaker: {self.speaker}")
        
        self.validate_model()

        logger.success(f"Model changed to: {model_id}")
        logger.info(f"Available speakers for the new model: {', '.join(self.get_available_speakers())}")
        logger.info(f"Sample rate set to the highest available: {self.sample_rate}")

    
    def change_speaker(self, speaker):
        # Only validate speaker if the model has a speakers attribute
        if hasattr(self.tts_model, 'speakers') and self.tts_model.speakers:
            if speaker not in self.get_available_speakers():
                logger.error(f"Speaker '{speaker}' is not supported for the current model '{self.model_id}'.")
                logger.info(f"Available speakers for this model: {', '.join(self.get_available_speakers())}")
                return
        else:
            # For models without speakers attribute, we can't validate, so we just log and proceed
            logger.info(f"Model doesn't expose speakers list, setting speaker to: {speaker}")
        
        self.speaker = speaker
        logger.success(f"Speaker changed to: {speaker}")


    def change_sample_rate(self, sample_rate):
        available_sample_rates = self.get_available_sample_rates()

        if sample_rate not in available_sample_rates:
            logger.error(f"Sample rate {sample_rate} is not supported for the current model '{self.model_id}'.")
            logger.info(f"Available sample rates for this model: {', '.join(map(str, available_sample_rates))}")
            return

        self.sample_rate = sample_rate
        logger.success(f"Sample rate changed to: {sample_rate}")

    def init_model(self):
        """
        Initialize the TTS model with proper error handling and resource management.
        """
        logger.info(f"Initializing model '{self.model_id}' for language '{self.language}'")

        # Set device properly
        if not torch.cuda.is_available() and self.device == "auto":
            self.device = 'cpu'
        if torch.cuda.is_available() and (self.device == "auto" or self.device == "cuda"):
            torch_dev = torch.device("cuda", 0)
            gpus_count = torch.cuda.device_count()
            logger.info(f"Using {gpus_count} GPU(s)...")
        else:
            torch_dev = torch.device(self.device)
        torch.set_num_threads(self.num_threads)

        # Create silero_models directory
        silero_models_dir = os.path.join(os.path.dirname(__file__), 'silero_models')
        if not os.path.exists(silero_models_dir):
            os.makedirs(silero_models_dir)

        # Get package URL from models config
        try:
            model_config = self.models_config['tts_models'][self.language][self.model_id]['latest']
            package_url = model_config['package']
        except KeyError as e:
            logger.error(f"Model configuration not found for language '{self.language}', model '{self.model_id}': {e}")
            raise ValueError(f"Model configuration not found for language '{self.language}', model '{self.model_id}'")

        # Define model file path
        model_file_name = f"{self.model_id}_{self.language}.pt"
        model_file_path = os.path.join(silero_models_dir, model_file_name)

        # Download model file if not exists
        if not os.path.exists(model_file_path):
            logger.info(f"Downloading model from {package_url} to {model_file_path}")
            try:
                import time
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = requests.get(package_url, stream=True, timeout=60)
                        if response.status_code == 200:
                            with open(model_file_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:  # Filter out keep-alive chunks
                                        f.write(chunk)
                            logger.success(f"Model downloaded successfully.")
                            break
                        else:
                            logger.warning(f"Failed to download model file. Status code: {response.status_code}. Attempt {attempt + 1}/{max_retries}")
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Network error during model download: {str(e)}. Attempt {attempt + 1}/{max_retries}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            logger.error(f"Failed to download model file after {max_retries} attempts")
                            raise Exception(f"Failed to download model file after {max_retries} attempts")
            except Exception as e:
                logger.error(f"Error downloading model: {str(e)}")
                raise

        # Load model from local file
        logger.info("Loading model")
        t1 = timeit.default_timer()
        try:
            from torch.package import PackageImporter
            model = PackageImporter(model_file_path).load_pickle("tts_models", "model")
            model.to(torch_dev)
            logger.info(f"Model to device takes {timeit.default_timer() - t1:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading model from {model_file_path}: {str(e)}")
            raise

        if torch.cuda.is_available() and (self.device == "auto" or self.device == "cuda"):
            logger.info("Synchronizing CUDA")
            t2 = timeit.default_timer()
            torch.cuda.synchronize()
            logger.info(f"Cuda Synch takes {timeit.default_timer() - t2:.2f} seconds")
        logger.success("Model is loaded")

        # Add a 'speakers' attribute if it doesn't exist, for compatibility with multi_v2 models
        if not hasattr(model, 'speakers'):
            # For models that don't have a speakers attribute, we need to determine available speakers
            # This is usually for newer model formats like TTSModelAcc_v2
            if hasattr(model, 'apply_tts') and hasattr(model, 'speakers_list'):
                # Some models have speakers_list instead of speakers
                model.speakers = model.speakers_list
            elif hasattr(model, 'speaker_manager') and hasattr(model.speaker_manager, 'speaker_ids'):
                # Some models have speaker_manager with speaker_ids
                model.speakers = list(model.speaker_manager.speaker_ids.keys())
            else:
                # Default case - for models that support multiple speakers but don't expose them directly
                # We'll set a default list of common silero speakers for this case
                # This can be overridden by the user if needed
                if self.language == 'ru':
                    model.speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']
                elif self.language == 'en':
                    model.speakers = ['en_0', 'en_1', 'en_2', 'en_3', 'en_4', 'en_5', 'en_6', 'en_7', 'en_8', 'en_9', 'en_10', 'random']
                else:
                    model.speakers = ['random']  # fallback for other languages

        return model
    @staticmethod

    def find_char_positions(self, string: str, char: str) -> list:
        pos = []  # list to store positions for each 'char' in 'string'
        for n in range(len(string)):
            if string[n] == char:
                pos.append(n)
        return pos

    @staticmethod
    def find_max_char_position(self, positions: list, limit: int) -> int:
        max_position = 0
        for pos in positions:
            if pos < limit:
                max_position = pos
            else:
                break
        return max_position

    @staticmethod
    def find_split_position(self, line: str, old_position: int, char: str, limit: int) -> int:
        positions = self.find_char_positions(line, char)
        new_position = self.find_max_char_position(positions, limit)
        position = max(new_position, old_position)
        return position

    def spell_digits(self, line) -> str:
        digits = re.findall(r'\d+', line)
        # Sort digits from largest to smallest - else "1 11" will be "один один один" but not "один одиннадцать"
        digits = sorted(digits, key=len, reverse=True)
        for digit in digits:
            line = line.replace(digit, self.converter.convert(int(digit[:12])))
        return line

    def preprocess_text(self, text):
        logger.info("Preprocessing text")

        if lang_data[self.language]['script'] == 'cyrillic' and is_latin(text):
            text = reverse_transliterate(text, self.language)
        elif lang_data[self.language]['script'] == 'latin' and is_cyrillic(text):
            if self.language in ["en", "fr", "es", "de"]:
                text = reverse_transliterate(text, self.language)
            else:
                text = transliterate(text, self.language)

        lines = text.split('\n')
        preprocessed_lines = []
        for line in lines:
            line = line.strip()  # Remove leading/trailing spaces
            if line == '':
                continue

            # Replace chars not supported by model
            for replacement in lang_data[self.language]['replacements']:
                line = line.replace(replacement[0], replacement[1])

            for pattern in lang_data[self.language]['patterns']:
                line = re.sub(pattern[0], pattern[1], line)

            line = self.spell_digits(line)

            preprocessed_lines.append(line)

        return preprocessed_lines

    def spell_digits(self, line: str, lang: str = None) -> str:
        """Enhanced digit spelling based on language."""
        if lang is None:
            lang = self.language
        if lang == 'ru':
            try:
                digits = re.findall(r'\d+', line)
                # Sort digits from largest to smallest
                digits = sorted(digits, key=len, reverse=True)
                for digit in digits:
                     # Limit to 12 digits for num2t4ru
                     line = line.replace(str(digit), self.converter.convert(int(digit[:12])), 1) # Replace one at a time to avoid overlap issues
            except (ValueError, IndexError):
                logger.warning(f"Warning: Could not convert digit '{digit[:12]}' to text, leaving as-is.")
                # If num2t4ru fails, leave the digit as is
                pass
        else:
            # For other languages, basic handling (could be expanded)
            # Example: Replace percentages, decimals, etc., as needed per language
            line = re.sub(r'(\d+)\.(\d+)', r'\1 point \2', line) # Basic decimal handling for non-Russian
        return line

    def preprocess_text(self, text, length_limit: int = None):
        if length_limit is None:
            length_limit = self.LINE_LENGTH_LIMIT
            
        logger.info(f"Preprocessing text with line length limit={length_limit}")
        if length_limit > 3:
            length_limit = length_limit - 2  # Keep a room for trailing char and '\n' char
        else:
            logger.error(f"ERROR: line length limit must be >= 3, got {length_limit}")
            raise ValueError(f"line length limit must be >= 3, got {length_limit}")

        preprocessed_text_len = 0
        preprocessed_lines = []
        
        # Split text into lines
        lines = text.split('\n') if isinstance(text, str) else text
        
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
            line = self.spell_digits(line, self.language)

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
                logger.warning(f"Warning: Forcing split at line {line_num+1} at char {length_limit}. This might cause unnatural breaks.")
                part = line[:length_limit] + "\n"
                preprocessed_lines.append(part)
                preprocessed_text_len += len(part)
                line = line[length_limit:].lstrip() # Remove leading spaces after forced split

        return preprocessed_lines, preprocessed_text_len

    def tts(self, text, output_file, progress_callback=None):
        """
        Generate speech from text and save to output file with enhanced error handling and progress tracking.
        
        Args:
            text (str): Text to convert to speech
            output_file (str): Path to output audio file
            progress_callback (callable, optional): Function to call with progress updates
        """
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")
        if not output_file or not isinstance(output_file, str):
            raise ValueError("output_file must be a non-empty string")
            
        # Preprocess text with length limit
        preprocessed_lines, preprocessed_text_len = self.preprocess_text(text)
        
        # Initialize Stats object
        s = self.Stats(preprocessed_text_len)
        audio_size = self.wave_header_size
        wave_file_number = 0
        current_line_idx = 0

        # Generate initial output filename
        base_output_file = output_file.rsplit('.', 1)[0]  # Remove extension
        output_filename = f'{base_output_file}_{self.speaker}_{wave_file_number}.wav'
        wf = self.init_wave_file(output_filename)
        logger.info(f"Writing to: {output_filename}")

        logger.info("Starting TTS")
        # Process each preprocessed line
        for line in preprocessed_lines:
            if not line.strip(): # Skip empty lines after stripping
                continue

            # Progress and status print
            if current_line_idx % 10 == 0: # Print every 10 lines or adjust frequency
                 progress_info = (
                     f'Line {current_line_idx+1}/{len(preprocessed_lines)} | '
                     f'Done: {s.done_percent:.1f}% | '
                     f'Time: {s.run_time}/{s.run_time_est} | '
                     f'TTS: {s.tts_time_current}/{s.tts_time_est} | '
                     f'File: {s.wave_mib}/{s.wave_mib_est} MiB'
                 )
                 logger.info(progress_info)
                 if progress_callback:
                     progress_callback(s)

            try:
                # --- Apply TTS ---
                audio = self.tts_model.apply_tts(
                    text=line.strip(), # Ensure no leading/trailing newlines passed to TTS
                    speaker=self.speaker,
                    sample_rate=self.sample_rate,
                    put_accent=self.put_accent,
                    put_yo=self.put_yo
                )

                # --- Handle Audio Output ---
                if audio is not None and len(audio) > 0:
                    audio_int16 = (audio * 32767).numpy().astype('int16')
                    audio_bytes = audio_int16.tobytes()
                    chunk_size_bytes = len(audio_bytes)

                    # Check if adding this chunk exceeds the file size limit
                    if audio_size + chunk_size_bytes > self.WAVE_FILE_SIZE_LIMIT:
                        logger.info(f"File size limit ({self.WAVE_FILE_SIZE_LIMIT} bytes) reached. Starting new file...")
                        wf.close()
                        s.next_file()
                        wave_file_number += 1
                        audio_size = self.wave_header_size # Reset size for new file
                        output_filename = f'{base_output_file}_{self.speaker}_{wave_file_number}.wav'
                        wf = self.init_wave_file(output_filename)
                        logger.info(f"Writing to: {output_filename}")

                    # Write the audio chunk
                    wf.writeframes(audio_bytes)
                    audio_size += chunk_size_bytes
                    s.update(line, chunk_size_bytes) # Update stats with actual bytes written
                else:
                    logger.warning(f"Warning: TTS returned no audio for line: {line[:50]}...") # Log short snippet
                    s.update(line, 0) # Still update stats for progress, even if no audio

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                     logger.warning(f"RuntimeError (likely OOM): {e}")
                     logger.info(f"Consider reducing sample_rate, using 'cpu' device, or simplifying the input line: {line[:50]}...")
                     # Optionally, try to continue with next line or exit gracefully
                     s.update(line, 0) # Update stats and continue
                else:
                     logger.error(f"RuntimeError during TTS for line: {line[:50]}... Error: {e}")
                     # Decide: continue, exit, or try fallback (e.g., skip line)
                     s.update(line, 0) # For now, update and continue
            except ValueError as e:
                 logger.error(f"ValueError during TTS for line: {line[:50]}... Error: {e}")
                 # Handle potential model-specific errors (e.g., unsupported characters after preprocessing)
                 s.update(line, 0) # Update stats and continue
            except Exception as e:
                 logger.error(f"Unexpected error during TTS for line: {line[:50]}... Error: {e}")
                 s.update(line, 0) # Update stats and continue

            current_line_idx += 1

        # Close the final wave file
        wf.close()
        logger.success(f'Speech saved to {output_filename}')
        sf_file, sf_samplerate = sf.read(output_filename)
        sd.play(data=sf_file, samplerate=self.sample_rate, blocking=True)
        sd.wait()
        print(f"Wave export file playing: {output_filename}...")
        return output_filename

    def init_wave_file(self, path):
        logger.info(f'Initializing wave file: {path}')
        wf = wave.open(path, 'wb')
        wf.setnchannels(1) #(self.wave_channels)
        wf.setsampwidth(2) #(self.wave_sample_width)
        wf.setframerate(self.sample_rate)
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
            self.sample_rate = 48000 #for default
            self.wave_channels = 1
            self.wave_sample_width = 2
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
            tts_time_s = int((self.wave_data_total / self.wave_channels / self.wave_sample_width / self.sample_rate))
            tts_time_est_s = int((tts_time_s * self.preprocessed_text_len / max(self.processed_text_len, 1))) # Avoid div by 0
            self.tts_time = str(timedelta(seconds=tts_time_s))
            self.tts_time_est = str(timedelta(seconds=tts_time_est_s))
            tts_time_current_s = int((self.wave_data_current / self.wave_channels / self.wave_sample_width / self.sample_rate))
            self.tts_time_current = str(timedelta(seconds=tts_time_current_s))

        def next_file(self):
            self.wave_data_current = 0

    def from_file(self, text_path, output_path, progress_callback=None):
        # Метод для генерации речи из текстового файла
        logger.info(f'Generating speech from file: {text_path}')
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.tts(text, output_path, progress_callback=progress_callback)

    @staticmethod
    def get_available_models():
        models_file = os.path.join(os.path.dirname(__file__), 'latest_silero_models.yml')

        if not os.path.exists(models_file):
            logger.warning(f"Models config file not found: {models_file}. Downloading...")
            SileroTTS.download_models_config_static(models_file)

        with open(models_file, 'r', encoding='utf-8') as f:
            models_config = yaml.safe_load(f)

        models_dict = {}
        for lang, models in models_config['tts_models'].items():
            models_dict[lang] = list(models.keys())

        return models_dict

    @staticmethod
    def get_latest_model(language):
        models_file = os.path.join(os.path.dirname(__file__), 'latest_silero_models.yml')

        if not os.path.exists(models_file):
            logger.warning(f"Models config file not found: {models_file}. Downloading...")
            SileroTTS.download_models_config_static(models_file)

        with open(models_file, 'r', encoding='utf-8') as f:
            models_config = yaml.safe_load(f)

        models = models_config['tts_models'][language]
        latest_model = sorted(models.keys(), reverse=True)[0]
        return latest_model

    @staticmethod
    def get_available_languages():
        models_file = os.path.join(os.path.dirname(__file__), 'latest_silero_models.yml')

        if not os.path.exists(models_file):
            logger.warning(f"Models config file not found: {models_file}. Downloading...")
            SileroTTS.download_models_config_static(models_file)

        with open(models_file, 'r', encoding='utf-8') as f:
            models_config = yaml.safe_load(f)

        return list(models_config['tts_models'].keys())
            
    
    @staticmethod
    def download_models_config_static(models_file=None):
        url = "https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml"
        response = requests.get(url)
        
        if models_file is None:
            models_file = os.path.join(os.path.dirname(__file__), 'models.yml')

        if response.status_code == 200:
            with open(models_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            logger.success(f"Models config file downloaded: {models_file}")
        else:
            logger.error(f"Failed to download models config file. Status code: {response.status_code}")
            raise Exception(f"Failed to download models config file. Status code: {response.status_code}")


    @staticmethod
    def get_available_sample_rates_static(language, model_id):
        models_file = os.path.join(os.path.dirname(__file__), 'latest_silero_models.yml')

        if not os.path.exists(models_file):
            logger.warning(f"Models config file not found: {models_file}. Downloading...")
            SileroTTS.download_models_config_static(models_file)

        with open(models_file, 'r', encoding='utf-8') as f:
            models_config = yaml.safe_load(f)

        model_config = models_config['tts_models'][language][model_id]['latest']
        sample_rates = model_config.get('sample_rate', [])

        if not isinstance(sample_rates, list):
            sample_rates = [sample_rates]

        return sample_rates


if __name__== '__main__':
    tts = SileroTTS(model_id='v4_ru',
                    language='ru',
                    speaker='kseniya_v2',  # Updated to work with multi_v2 models
                    sample_rate=48000,
                    device='cpu')

    # Speech generation from text
    # text = "Проверка Silero"
    # tts.tts(text, 'output.wav')

    try:
        logger.info(f"Available speakers for model {tts.model_id}: {tts.get_available_speakers()}")
    except Exception as e:
        logger.warning(f"Could not get available speakers: {e}")

    # Generating speech from a file
    # tts.from_file('input.txt', 'output.wav')

    logger.info(f"Available models: {SileroTTS.get_available_models()}")
