# Changelog

All notable changes to SileroTTS.

## [0.8.0] - 2026-05-09

### Added
- **Analytics Dashboard** - Comprehensive server monitoring and reachability analysis
  - Real-time metrics display (uptime, response time, success rate)
  - Interactive charts (Chart.js based)
  - System load monitoring (CPU, memory visualization)
  - Response time trends and distribution analysis
  - Endpoint performance tracking
  - Error analysis with timeline visualization
  - Request volume statistics by hour
  - Health status indicators
  - Recent checks table
- **Analytics API Endpoints**
  - `GET /api/analytics` - Get all analytics data
  - `POST /api/analytics/check` - Record reachability check
  - `GET /api/analytics/export` - Export data (JSON/CSV)
  - `POST /api/analytics/import` - Import session data
  - `DELETE /api/analytics` - Clear analytics
- **Session Management**
  - Export analytics to JSON format
  - Export analytics to CSV format
  - Import previous sessions for comparison
  - Clear session data
  - Automatic localStorage backup
- **Automatic Request Tracking**
  - All TTS endpoints automatically tracked
  - Response time measurement
  - Success/failure status recording
  - HTTP status code logging
  - Error details capture
- **Data Visualization**
  - Reachability gauge chart
  - Load trend chart
  - Response time bar chart
  - Status distribution pie chart
  - Hourly volume line chart
  - Error analysis bar chart
  - Response distribution doughnut chart
- **Dashboard Features**
  - Auto-refresh every 30 seconds
  - Manual refresh button
  - Last updated timestamp
  - Timeline of events
  - Health status cards
  - Endpoint performance table
  - Recent checks table

### Changed
- Updated `/data` endpoint to serve analytics dashboard
- Enhanced `api_server.py` with analytics tracking
- Improved error handling with automatic tracking
- Added metrics calculation and aggregation
- Updated README with API server documentation
- Enhanced version to 0.8.0

### Fixed
- Fixed analytics data persistence
- Fixed metrics calculation for empty datasets
- Fixed auto-refresh functionality
- Fixed export/import data handling

### Security
- All analytics data stored locally
- No external data transmission
- CORS configured for browser access

## [0.7.2] - Previous version

### Added
- Basic API server functionality
- Web UI with modern design
- Multiple TTS endpoints (audio, stream, json)
- Model and speaker management
- History and cache management
- Settings persistence

### Features
- FastAPI-based REST API
- Jinja2 templates for web UI
- Base64 audio encoding
- Direct audio file serving
- Streaming support
- CORS middleware
- Audio file caching
- Generation history

## [0.1.0] - 2025-01-24

### Added
- Input validation for all parameters in constructor and methods
- Network resilience with retry mechanisms and timeouts
- Comprehensive error handling with informative messages
- KeyboardInterrupt handling in CLI
- Proper resource management
- Batch processing error handling
- Empty audio handling in TTS method
- Missing speaker validation
- Documentation for all methods

### Changed
- Updated version from 0.0.5 to 0.1.0 (stable release)
- Improved model download with retry logic (3 attempts with exponential backoff)
- Enhanced config download with timeout and retry mechanisms
- Better error messages throughout the codebase
- Updated README to reflect v0.1 stability features
- Improved CLI error handling and user feedback

### Fixed
- Fixed potential crashes when no speakers are available
- Fixed empty audio handling that could cause write errors
- Fixed network timeout issues during model/config downloads
- Fixed batch processing to continue when individual files fail
- Fixed resource management in model loading

### Removed
- Removed redundant network requests without proper error handling
- Removed unsafe file operations without validation

## [0.0.5] - Previous version

Initial unstable version with basic functionality.