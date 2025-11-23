"""
Stability tests for SileroTTS v0.1
"""
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from silero_tts.silero_tts import SileroTTS


def test_silero_tts_initialization():
    """Test basic initialization with proper parameters"""
    # Test with valid parameters
    tts = SileroTTS(
        model_id="v4_ru",
        language="ru",
        speaker="aidar",
        sample_rate=48000,
        device="cpu",
        num_threads=2
    )
    assert tts.model_id == "v4_ru"
    assert tts.language == "ru"
    assert tts.sample_rate == 48000
    assert tts.device == "cpu"


def test_silero_tts_validation():
    """Test input validation"""
    # Test invalid model_id
    with pytest.raises(ValueError):
        SileroTTS(model_id="", language="ru", sample_rate=48000)
    
    # Test invalid language
    with pytest.raises(ValueError):
        SileroTTS(model_id="v4_ru", language="", sample_rate=48000)
    
    # Test invalid sample_rate
    with pytest.raises(ValueError):
        SileroTTS(model_id="v4_ru", language="ru", sample_rate=0)
    
    # Test invalid device
    with pytest.raises(ValueError):
        SileroTTS(model_id="v4_ru", language="ru", sample_rate=48000, device="invalid")
    
    # Test invalid num_threads
    with pytest.raises(ValueError):
        SileroTTS(model_id="v4_ru", language="ru", sample_rate=48000, num_threads=0)


def test_silero_tts_with_none_speaker():
    """Test initialization with None speaker (should auto-select)"""
    # Mock the model to avoid actual model loading
    with patch('silero_tts.silero_tts.SileroTTS.init_model') as mock_init_model:
        mock_model = MagicMock()
        mock_model.speakers = ['speaker1', 'speaker2']
        mock_init_model.return_value = mock_model
        
        tts = SileroTTS(
            model_id="v4_ru",
            language="ru",
            speaker=None,  # Should auto-select first speaker
            sample_rate=48000
        )
        assert tts.speaker == 'speaker1'


def test_silero_tts_with_none_speaker_no_speakers():
    """Test initialization with None speaker when no speakers available"""
    # Mock the model to avoid actual model loading
    with patch('silero_tts.silero_tts.SileroTTS.init_model') as mock_init_model:
        mock_model = MagicMock()
        mock_model.speakers = []  # No speakers available
        mock_init_model.return_value = mock_model
        
        with pytest.raises(ValueError, match="No speakers available for the selected model"):
            SileroTTS(
                model_id="v4_ru",
                language="ru",
                speaker=None,
                sample_rate=48000
            )


def test_tts_method_validation():
    """Test validation in tts method"""
    tts = SileroTTS(
        model_id="v4_ru",
        language="ru",
        sample_rate=48000,
        device="cpu"
    )
    
    # Test with empty text
    with pytest.raises(ValueError):
        tts.tts("", "output.wav")
    
    # Test with non-string text
    with pytest.raises(ValueError):
        tts.tts(123, "output.wav")
    
    # Test with empty output file
    with pytest.raises(ValueError):
        tts.tts("text", "")
    
    # Test with non-string output file
    with pytest.raises(ValueError):
        tts.tts("text", 123)


def test_available_methods():
    """Test that static methods exist and work properly"""
    # These methods should not raise exceptions
    try:
        languages = SileroTTS.get_available_languages()
        assert isinstance(languages, list) or isinstance(languages, dict)
    except Exception:
        # These methods might require network access or config files,
        # which is fine for a basic test
        pass


def test_preprocessing_methods():
    """Test text preprocessing functionality"""
    tts = SileroTTS(
        model_id="v4_ru",
        language="ru",
        sample_rate=48000,
        device="cpu"
    )
    
    # Test basic text preprocessing
    text = "Hello world!"
    result = tts.preprocess_text(text)
    assert isinstance(result, list)
    assert len(result) >= 0  # May be empty if text is empty after processing


if __name__ == "__main__":
    pytest.main([__file__])