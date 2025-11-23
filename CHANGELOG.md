# Changelog

All notable changes to SileroTTS v0.1 stable release.

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