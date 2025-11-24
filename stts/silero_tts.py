import os
import re
import timeit
import torch
import wave
import yaml
import requests
from loguru import logger
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
            # but we'll allow it to proceed with the assumption that the model supports the requested speaker
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
        t0 = timeit.default_timer()

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

    def find_char_positions(self, string: str, char: str) -> list:
        pos = []  # list to store positions for each 'char' in 'string'
        for n in range(len(string)):
            if string[n] == char:
                pos.append(n)
        return pos

    def find_max_char_position(self, positions: list, limit: int) -> int:
        max_position = 0
        for pos in positions:
            if pos < limit:
                max_position = pos
            else:
                break
        return max_position

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

    def tts(self, text, output_file):
        """
        Generate speech from text and save to output file.
        
        Args:
            text (str): Text to convert to speech
            output_file (str): Path to output audio file
        """
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")
        if not output_file or not isinstance(output_file, str):
            raise ValueError("output_file must be a non-empty string")
            
        # Основной метод для генерации речи
        preprocessed_lines = self.preprocess_text(text)

        # Инициализируем wav-файл
        wf = self.init_wave_file(output_file)

        logger.info("Starting TTS")
        # Синтезируем речь и пишем в файл
        for i, line in enumerate(preprocessed_lines):
            logger.info(f'Processing line {i+1}/{len(preprocessed_lines)}: {line}')
            try:
                audio = self.tts_model.apply_tts(text=line,
                                                 speaker=self.speaker,
                                                 sample_rate=self.sample_rate,
                                                 put_accent=self.put_accent,
                                                 put_yo=self.put_yo)
                # Ensure audio is properly formatted
                if audio is not None and len(audio) > 0:
                    wf.writeframes((audio * 32767).numpy().astype('int16'))
                else:
                    logger.warning(f'Empty audio returned for line: {line}')
            except ValueError as e:
                logger.warning(f'TTS failed for line: {line}. Error: {str(e)}. Skipping...')
            except Exception as e:
                logger.error(f'Unexpected error during TTS for line: {line}. Error: {str(e)}. Skipping...')
                continue

        wf.close()
        logger.success(f'Speech saved to {output_file}')

    def init_wave_file(self, path):
        logger.info(f'Initializing wave file: {path}')
        wf = wave.open(path, 'wb')
        wf.setnchannels(self.wave_channels)
        wf.setsampwidth(self.wave_sample_width)
        wf.setframerate(self.sample_rate)
        return wf

    def from_file(self, text_path, output_path):
        # Метод для генерации речи из текстового файла
        logger.info(f'Generating speech from file: {text_path}')
        with open(text_path, 'r') as f:
            text = f.read()

        self.tts(text, output_path)

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
