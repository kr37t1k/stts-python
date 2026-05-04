// State
const state = {
    endpoint: 'audio',
    models: {},
    languages: [],
    speakers: [],
    sampleRates: [],
    speakersData: {},
    sampleRatesData: {},
    currentAudioUrl: null,
    currentAudioId: null,
    isGenerating: false,
    history: []
};

// Storage keys
const STORAGE_KEYS = {
    SETTINGS: 'sileroTTS_settings',
    HISTORY: 'sileroTTS_history'
};

// DOM Elements
const elements = {
    text: document.getElementById('text'),
    language: document.getElementById('language'),
    model: document.getElementById('model'),
    speaker: document.getElementById('speaker'),
    device: document.getElementById('device'),
    sampleRate: document.getElementById('sampleRate'),
    sampleRateValue: document.getElementById('sampleRateValue'),
    generateBtn: document.getElementById('generateBtn'),
    status: document.getElementById('status'),
    audioPlayer: document.getElementById('audioPlayer'),
    audioElement: document.getElementById('audioElement'),
    audioFormat: document.getElementById('audioFormat'),
    audioDuration: document.getElementById('audioDuration'),
    downloadBtn: document.getElementById('downloadBtn'),
    copyUrlBtn: document.getElementById('copyUrlBtn'),
    baseUrl: document.getElementById('baseUrl'),
    putAccent: document.getElementById('putAccent'),
    putYo: document.getElementById('putYo'),
    autoPlay: document.getElementById('autoPlay'),
    numThreads: document.getElementById('numThreads'),
    threadsValue: document.getElementById('threadsValue'),
    applySettingsBtn: document.getElementById('applySettingsBtn'),
    refreshModelsBtn: document.getElementById('refreshModelsBtn'),
    clearCacheBtn: document.getElementById('clearCacheBtn'),
    historyContainer: document.getElementById('historyContainer'),
    clearHistoryBtn: document.getElementById('clearHistoryBtn'),
    audioPlayerPlayPause: document.getElementById('audioPlayerPlayPause'),
    audioPlayerProgress: document.getElementById('audioPlayerProgress'),
    audioPlayerCurrentTime: document.getElementById('audioPlayerCurrentTime'),
    audioPlayerWaveform: document.getElementById('audioPlayerWaveform')
};

// Initialize
async function init() {
    showStatus('Loading models...', 'loading');
    loadSettings();
    await loadModels();
    setupEventListeners();
    setupAudioPlayer();
    updateCurlExample();
    updateJsExample();
    await loadHistoryFromServer();
    hideStatus();
}

// Load settings from localStorage
function loadSettings() {
    try {
        const saved = localStorage.getItem(STORAGE_KEYS.SETTINGS);
        if (saved) {
            const settings = JSON.parse(saved);
            if (settings.putAccent) elements.putAccent.classList.add('active');
            if (settings.putYo) elements.putYo.classList.add('active');
            if (settings.autoPlay !== false) elements.autoPlay.classList.add('active');
            if (settings.numThreads) {
                elements.numThreads.value = settings.numThreads;
                elements.threadsValue.textContent = settings.numThreads;
            }
            if (settings.baseUrl) {
                elements.baseUrl.value = settings.baseUrl;
            }
        }
    } catch (error) {
        console.warn('Failed to load settings:', error);
    }
}

// Save settings to localStorage
function saveSettings() {
    try {
        const settings = {
            putAccent: elements.putAccent.classList.contains('active'),
            putYo: elements.putYo.classList.contains('active'),
            autoPlay: elements.autoPlay.classList.contains('active'),
            numThreads: parseInt(elements.numThreads.value),
            baseUrl: elements.baseUrl.value
        };
        localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
    } catch (error) {
        console.warn('Failed to save settings:', error);
    }
}

// Load history from localStorage
function loadHistory() {
    // History is now loaded from server, not localStorage
    state.history = [];
}

// Save history to localStorage
function saveHistory() {
    // History is now stored on server, not localStorage
}

// Add to history (local temporary entry)
function addToHistory(audioUrl, text, duration, sampleRate, language, model, speaker) {
    const id = Date.now().toString();
    const entry = {
        id,
        timestamp: Date.now(),
        text,
        duration: duration || 0,
        sampleRate: sampleRate || parseInt(elements.sampleRate.value),
        language,
        model,
        speaker,
        audioUrl // Will be replaced with server URL
    };

    state.history.unshift(entry);
    
    // Keep only last 50 entries
    if (state.history.length > 50) {
        state.history = state.history.slice(0, 50);
    }
    
    renderHistory();
    return id;
}

// Load history from server
async function loadHistoryFromServer() {
    try {
        const response = await fetch(`${elements.baseUrl.value}/history`);
        const data = await response.json();
        state.history = data.history || [];
        renderHistory();
    } catch (error) {
        console.warn('Failed to load history from server:', error);
    }
}

// Load available models from API
async function loadModels() {
    try {
        const langParam = elements.language.value ? `?language=${elements.language.value}` : '';
        const response = await fetch(`${elements.baseUrl.value}/models${langParam}`);
        const data = await response.json();
        
        state.models = data.available_models || {};
        state.languages = data.available_languages || [];
        state.speakersData = data.available_speakers || {};
        state.sampleRatesData = data.available_sample_rates || {};
        
        console.log('Loaded models:', state.models);
        console.log('Loaded speakersData:', state.speakersData);
        console.log('Loaded sampleRatesData:', state.sampleRatesData);
        
        updateLanguageSelect();
        
        if (state.languages.length > 0 && !elements.language.value) {
            elements.language.value = state.languages[0];
            await updateModelSelect();
        }
    } catch (error) {
        showStatus('Failed to load models: ' + error.message, 'error');
    }
}

// Update language select
function updateLanguageSelect() {
    const currentLang = elements.language.value;
    elements.language.innerHTML = state.languages.map(lang => 
        `<option value="${lang}" ${lang === currentLang ? 'selected' : ''}>${getLanguageName(lang)}</option>`
    ).join('');
    updateModelSelect();
}

// Update model select based on language
async function updateModelSelect() {
    const lang = elements.language.value;
    const models = state.models[lang] || [];
    
    if (models.length === 0) {
        elements.model.innerHTML = '<option value="">No models available</option>';
        elements.speaker.innerHTML = '<option value="">Select model first</option>';
        elements.sampleRate.disabled = true;
        return;
    }

    elements.sampleRate.disabled = false;
    elements.model.innerHTML = models.map(model => 
        `<option value="${model}">${model}</option>`
    ).join('');
    
    // Set first model as selected by default
    if (models.length > 0) {
        elements.model.value = models[0];
        await loadSpeakersFromServer(lang);
        updateSampleRates();
    }
}

// Load speakers from server based on language and model
async function loadSpeakersFromServer(lang) {
    const modelId = elements.model.value;
    
    if (!modelId) {
        elements.speaker.innerHTML = '<option value="">Select model first</option>';
        return;
    }

    const cacheKey = `${lang}_${modelId}`;
    
    if (state.speakersData[cacheKey]) {
        state.speakers = state.speakersData[cacheKey].speakers;
        elements.speaker.innerHTML = state.speakers.map(s => 
            `<option value="${s}">${s}</option>`
        ).join('');
        elements.speaker.value = state.speakers[0];
        
        if (state.speakersData[cacheKey].sample_rates) {
            state.sampleRatesData[lang] = state.sampleRatesData[lang] || {};
            state.sampleRatesData[lang][modelId] = state.speakersData[cacheKey].sample_rates;
            updateSampleRates();
        }
        return;
    }

    try {
        const response = await fetch(`${elements.baseUrl.value}/speakers?language=${lang}&model_id=${modelId}`);
        const data = await response.json();
        
        if (data.speakers) {
            state.speakers = data.speakers.speakers;
            state.speakersData[cacheKey] = {
                speakers: data.speakers.speakers,
                sample_rates: data.speakers.sample_rates
            };
            
            if (data.speakers.sample_rates) {
                state.sampleRatesData[lang] = state.sampleRatesData[lang] || {};
                state.sampleRatesData[lang][modelId] = data.speakers.sample_rates;
            }
            
            elements.speaker.innerHTML = state.speakers.map(s => 
                `<option value="${s}">${s}</option>`
            ).join('');
            elements.speaker.value = state.speakers[0];
            updateSampleRates();
        } else {
            elements.speaker.innerHTML = '<option value="random">random</option>';
            state.speakers = ['random'];
            console.warn(data.speakers);
            updateSampleRates();
        }
    } catch (error) {
        console.warn('Failed to load speakers from server, using fallback:', error);
        const fallbackSpeakers = {
            'ru': ['kseniya_v2', 'aidar_v2', 'baya_v2', 'random'],
            'en': ['en_0', 'en_1', 'lj', 'random'],
            'de': ['thorsten_v2', 'random'],
            'es': ['tux_v2', 'random'],
            'fr': ['gilles_v2', 'random'],
            'uk': ['mykyta_v2', 'random'],
            'uz': ['dilnavoz_v2', 'random'],
            'tt': ['dilyara', 'random'],
            'ba': ['aigul', 'random'],
            'xal': ['erdni', 'random'],
            'default': ['random']
        };
        state.speakers = fallbackSpeakers[lang] || fallbackSpeakers['default'];
        elements.speaker.innerHTML = state.speakers.map(s => 
            `<option value="${s}">${s}</option>`
        ).join('');
        updateSampleRates();
    }
}

// Update sample rates based on selected model
function updateSampleRates() {
    const modelId = elements.model.value;
    const lang = elements.language.value;
    
    if (!modelId) {
        elements.sampleRate.value = 48000;
        elements.sampleRateValue.textContent = '48000';
        return;
    }
    
    let defaultRates = [];
    if (state.sampleRatesData[lang] && state.sampleRatesData[lang][modelId]) {
        defaultRates = state.sampleRatesData[lang][modelId];
    }
    
    if (defaultRates.length === 0) {
        defaultRates = [8000, 16000, 24000, 48000];
        
        if (modelId.includes('_8khz')) {
            defaultRates = [8000];
        } else if (modelId.includes('_16khz')) {
            defaultRates = [8000, 16000];
        } else if (modelId.includes('v5') || modelId.includes('v4') || modelId.includes('v3')) {
            defaultRates = [8000, 24000, 48000];
        }
    }
    
    elements.sampleRate.min = Math.min(...defaultRates);
    elements.sampleRate.max = Math.max(...defaultRates);
    elements.sampleRate.step = Math.min(...defaultRates);
    
    const maxRate = Math.max(...defaultRates);
    if (elements.sampleRate.value < Math.min(...defaultRates) || 
        elements.sampleRate.value > Math.max(...defaultRates)) {
        elements.sampleRate.value = maxRate;
    }
    elements.sampleRateValue.textContent = elements.sampleRate.value;
}

// Get language display name
function getLanguageName(code) {
    const names = {
        'ru': 'Russian (Русский)',
        'en': 'English',
        'de': 'German (Deutsch)',
        'es': 'Spanish (Español)',
        'fr': 'French (Français)',
        'uk': 'Ukrainian (Українська)',
        'uz': 'Uzbek (O\'zbek)',
        'tt': 'Tatar (Татар)',
        'ba': 'Bashkir (Башҡорт)',
        'xal': 'Kalmyk (Хальмг)',
        'indic': 'Indic Languages',
        'cyrillic': 'Cyrillic',
        'multi': 'Multi-language'
    };
    return names[code] || code.toUpperCase();
}

// Setup event listeners
function setupEventListeners() {
    document.querySelectorAll('.endpoint-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.endpoint-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            state.endpoint = tab.dataset.endpoint;
            updateCurlExample();
            updateJsExample();
        });
    });

    elements.language.addEventListener('change', async () => {
        updateModelSelect();
        updateCurlExample();
        updateJsExample();
    });

    elements.model.addEventListener('change', async () => {
        await loadSpeakersFromServer(elements.language.value);
        updateCurlExample();
        updateJsExample();
    });

    elements.sampleRate.addEventListener('input', () => {
        elements.sampleRateValue.textContent = elements.sampleRate.value;
    });

    elements.sampleRate.addEventListener('change', () => {
        elements.sampleRateValue.textContent = elements.sampleRate.value;
    });

    document.querySelectorAll('.quick-btn[data-sample]').forEach(btn => {
        btn.addEventListener('click', () => {
            elements.text.value = btn.dataset.sample;
            btn.style.transform = 'scale(0.95)';
            setTimeout(() => btn.style.transform = '', 100);
        });
    });

    elements.generateBtn.addEventListener('click', generateSpeech);

    elements.downloadBtn.addEventListener('click', () => {
        if (state.currentAudioUrl) {
            const a = document.createElement('a');
            a.href = state.currentAudioUrl;
            a.download = `tts_${elements.language.value}_${Date.now()}.wav`;
            a.click();
        }
    });

    elements.copyUrlBtn.addEventListener('click', () => {
        if (state.currentAudioUrl) {
            navigator.clipboard.writeText(state.currentAudioUrl);
            showStatus('URL copied to clipboard!', 'success');
        }
    });

    document.querySelectorAll('.toggle').forEach(toggle => {
        toggle.addEventListener('click', () => {
            toggle.classList.toggle('active');
            toggle.dataset.value = toggle.classList.contains('active');
            saveSettings();
        });
    });

    elements.numThreads.addEventListener('input', () => {
        elements.threadsValue.textContent = elements.numThreads.value;
    });

    elements.applySettingsBtn.addEventListener('click', () => {
        applySettings();
        saveSettings();
    });
    
    elements.refreshModelsBtn.addEventListener('click', loadModels);
    elements.clearCacheBtn.addEventListener('click', clearCache);

    if (elements.clearHistoryBtn) {
        elements.clearHistoryBtn.addEventListener('click', clearHistory);
    }

    elements.baseUrl.addEventListener('change', () => {
        updateCurlExample();
        updateJsExample();
        saveSettings();
    });
}

// Setup audio player controls (SoundCloud style)
function setupAudioPlayer() {
    if (!elements.audioPlayerPlayPause || !elements.audioElement) return;
    
    const audio = elements.audioElement;
    const playPauseBtn = elements.audioPlayerPlayPause;
    const progress = elements.audioPlayerProgress;
    const currentTimeEl = elements.audioPlayerCurrentTime;
    
    // Play/Pause toggle
    playPauseBtn.addEventListener('click', () => {
        if (audio.paused) {
            audio.play();
        } else {
            audio.pause();
        }
    });
    
    // Update play/pause icon
    audio.addEventListener('play', () => {
        playPauseBtn.classList.add('playing');
    });
    
    audio.addEventListener('pause', () => {
        playPauseBtn.classList.remove('playing');
    });
    
    // Update progress bar
    audio.addEventListener('timeupdate', () => {
        if (progress && audio.duration) {
            const percent = (audio.currentTime / audio.duration) * 100;
            progress.style.width = `${percent}%`;
        }
        
        if (currentTimeEl) {
            currentTimeEl.textContent = formatTime(audio.currentTime);
        }
    });
    
    // Seek on click
    if (progress) {
        progress.parentElement.addEventListener('click', (e) => {
            const rect = progress.parentElement.getBoundingClientRect();
            const percent = (e.clientX - rect.left) / rect.width;
            audio.currentTime = percent * audio.duration;
        });
    }
    
    // Update duration when loaded
    audio.addEventListener('loadedmetadata', () => {
        if (elements.audioDuration) {
            elements.audioDuration.textContent = `Duration: ${formatTime(audio.duration)}`;
        }
    });
}

// Format time
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Clear history
async function clearHistory() {
    try {
        const response = await fetch(`${elements.baseUrl.value}/cache`, {
            method: 'DELETE'
        });
        const data = await response.json();
        showStatus(`History cleared! ${data.files_deleted} files deleted`, 'success');
        await loadHistoryFromServer();
    } catch (error) {
        showStatus('Error clearing history: ' + error.message, 'error');
    }
}

// Render history list
function renderHistory() {
    if (!elements.historyContainer) return;
    
    if (state.history.length === 0) {
        elements.historyContainer.innerHTML = `
            <div class="history-empty">
                <div class="history-empty-icon">📋</div>
                <p>No generations yet. Create your first speech!</p>
            </div>
        `;
        return;
    }
    
    elements.historyContainer.innerHTML = `
        <div class="history-header">
            <h3>Recent Generations (${state.history.length})</h3>
            <button class="btn btn-small btn-secondary" id="clearHistoryBtn">Clear All</button>
        </div>
        <div class="history-list">
            ${state.history.map(entry => `
                <div class="history-item" data-id="${entry.id}">
                    <div class="history-item-header">
                        <span class="history-time">${formatDateTime(new Date(entry.timestamp).getTime())}</span>
                        <span class="history-meta">${getLanguageName(entry.language)} • ${entry.model_id || entry.model || 'default'}</span>
                    </div>
                    <div class="history-item-text">${escapeHtml(entry.text)}</div>
                    <div class="history-item-player">
                        <audio class="sc-player" controls style="width: 100%; height: 50px;">
                            <source src="${entry.audio_url || ''}" type="audio/wav">
                            Your browser does not support the audio element.
                        </audio>
                        <div class="history-item-meta" style="margin-top: 0.5rem; font-size: 0.75rem; color: var(--text-muted);">
                            <span>Duration: ${formatTime(entry.duration || 0)}</span>
                            <span style="margin-left: 1rem;">Sample Rate: ${entry.sample_rate || entry.sampleRate || 'unknown'} Hz</span>
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

// Generate waveform bars
function generateWaveformBars() {
    let bars = '';
    for (let i = 0; i < 40; i++) {
        const height = Math.random() * 60 + 20;
        bars += `<div class="sc-waveform-bar" style="height: ${height}%"></div>`;
    }
    return bars;
}

// Format datetime
function formatDateTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    
    return date.toLocaleDateString();
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Generate speech
async function generateSpeech() {
    if (state.isGenerating) return;
    
    const text = elements.text.value.trim();
    if (!text) {
        showStatus('Please enter some text to convert', 'error');
        return;
    }

    state.isGenerating = true;
    updateButtonState(true);
    hideAudioPlayer();

    const requestData = {
        text: text,
        language: elements.language.value,
        model_id: elements.model.value || undefined,
        speaker: elements.speaker.value || undefined,
        sample_rate: parseInt(elements.sampleRate.value),
        device: elements.device.value
    };

    try {
        let endpoint = '/tts';
        if (state.endpoint === 'audio') endpoint = '/tts/audio';
        else if (state.endpoint === 'stream') endpoint = '/tts/stream';

        showStatus('Generating speech...', 'loading');

        const response = await fetch(`${elements.baseUrl.value}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Generation failed');
        }

        let audioUrl, duration, sampleRate;
        
        if (state.endpoint === 'json') {
            const data = await response.json();
            const audioBytes = base64ToBlob(data.audio_data, 'audio/wav');
            audioUrl = URL.createObjectURL(audioBytes);
            duration = data.duration;
            sampleRate = data.sample_rate;
        } else {
            const blob = await response.blob();
            audioUrl = URL.createObjectURL(blob);
            sampleRate = parseInt(elements.sampleRate.value);
            
            // Get duration from audio
            const audio = new Audio();
            audio.onloadedmetadata = () => {
                duration = audio.duration;
            };
            audio.src = audioUrl;
        }

        showStatus('Speech generated successfully!', 'success');

        // Reload history from server
        await loadHistoryFromServer();
        
        showAudioPlayer(audioUrl, duration, sampleRate);

    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
    } finally {
        state.isGenerating = false;
        updateButtonState(false);
    }
}

// Update button state
function updateButtonState(loading) {
    elements.generateBtn.disabled = loading;
    document.getElementById('btnIcon').textContent = loading ? '' : '🔊';
    document.getElementById('btnText').textContent = loading ? 'Generating...' : 'Generate Speech';
    
    if (loading) {
        const spinner = document.createElement('div');
        spinner.className = 'spinner';
        spinner.id = 'btnSpinner';
        document.getElementById('btnIcon').appendChild(spinner);
    } else {
        const spinner = document.getElementById('btnSpinner');
        if (spinner) spinner.remove();
    }
}

// Show audio player
function showAudioPlayer(url, duration = null, sampleRate = null) {
    state.currentAudioUrl = url;
    elements.audioElement.src = url;
    
    if (elements.audioFormat) {
        if (sampleRate) {
            elements.audioFormat.textContent = `Sample Rate: ${sampleRate} Hz`;
        } else {
            elements.audioFormat.textContent = `Sample Rate: ${elements.sampleRate.value} Hz`;
        }
    }
    
    elements.audioPlayer.classList.add('visible');

    // Reset progress
    const progressFill = document.querySelector('.sc-progress-fill');
    if (progressFill) {
        progressFill.style.width = '0%';
    }
    
    if (elements.autoPlay.classList.contains('active')) {
        elements.audioElement.play().catch(e => console.log('Auto-play blocked:', e));
    }
}

// Hide audio player
function hideAudioPlayer() {
    elements.audioPlayer.classList.remove('visible');
    if (state.currentAudioUrl) {
        URL.revokeObjectURL(state.currentAudioUrl);
        state.currentAudioUrl = null;
    }
}

// Show status message
function showStatus(message, type) {
    elements.status.textContent = message;
    elements.status.className = `status visible status-${type}`;
}

// Hide status message
function hideStatus() {
    elements.status.classList.remove('visible');
}

// Apply settings
async function applySettings() {
    try {
        const response = await fetch(`${elements.baseUrl.value}/settings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                put_accent: elements.putAccent.classList.contains('active'),
                put_yo: elements.putYo.classList.contains('active'),
                num_threads: parseInt(elements.numThreads.value)
            })
        });

        const data = await response.json();
        showStatus('Settings applied successfully!', 'success');
        setTimeout(hideStatus, 3000);
    } catch (error) {
        showStatus('Error applying settings: ' + error.message, 'error');
    }
}

// Clear cache
async function clearCache() {
    try {
        const response = await fetch(`${elements.baseUrl.value}/cache`, {
            method: 'DELETE'
        });
        const data = await response.json();
        showStatus(`Cache cleared: ${data.files_deleted} files deleted`, 'success');
        setTimeout(hideStatus, 3000);
    } catch (error) {
        showStatus('Error clearing cache: ' + error.message, 'error');
    }
}

// Base64 to Blob
function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

// Update curl example
function updateCurlExample() {
    const lang = elements.language.value;
    const model = elements.model.value;
    const baseUrl = location.origin;
    const endpoint = state.endpoint === 'json' ? '/tts' : `/tts/${state.endpoint}`;
    
    document.getElementById('curlExample').innerHTML = 
`<span class="code-comment"># Generate audio (${state.endpoint})</span>
<span class="code-keyword">curl</span> -X POST <span class="code-string">"${baseUrl}${endpoint}"</span> \\
  -H <span class="code-string">"Content-Type: application/json"</span> \\
  -d <span class="code-string">'{
    "text": "Hello World",
    "language": "${lang}"${model ? `,
    "model_id": "${model}"` : ''}
  }'</span>${state.endpoint !== 'json' ? ' -o audio.wav' : ''}`;
}

// Update JS example
function updateJsExample() {
    const lang = elements.language.value;
    const model = elements.model.value;
    const baseUrl = location.origin;
    const endpoint = state.endpoint === 'json' ? '/tts' : `/tts/${state.endpoint}`;
    
    document.getElementById('jsExample').innerHTML = 
`<span class="code-comment">// JavaScript - ${state.endpoint} endpoint</span>
<span class="code-keyword">const</span> response = <span class="code-keyword">await</span> fetch(<span class="code-string">'${baseUrl}${endpoint}'</span>, {
  method: <span class="code-string">'POST'</span>,
  headers: { <span class="code-property">'Content-Type'</span>: <span class="code-string">'application/json'</span> },
  body: JSON.stringify({
    text: <span class="code-string">'Hello World'</span>,
    language: <span class="code-string">'${lang}'</span>${model ? `,
    model_id: <span class="code-string">'${model}'</span>` : ''}
  })
});
${state.endpoint === 'json' ? 
`<span class="code-keyword">const</span> data = <span class="code-keyword">await</span> response.json();
<span class="code-comment">// data.audio_data contains base64 audio</span>` : 
`<span class="code-keyword">const</span> blob = <span class="code-keyword">await</span> response.blob();
<span class="code-keyword">const</span> url = URL.createObjectURL(blob);`}`;
}

// Start the app
init();
