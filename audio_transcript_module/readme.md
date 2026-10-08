INPUT:
Video file (.mp4/.avi/.mov)

PROCESS:
1. Extract audio
2. WhisperX transcription
3. Word-level alignment

OUTPUT:
JSON containing:
- word
- start
- end
- transcript
- language
