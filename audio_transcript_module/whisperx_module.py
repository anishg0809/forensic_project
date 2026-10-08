import os
import json
import subprocess
import torch
import whisperx


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "large-v3"
BATCH_SIZE = 4


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"


# ============================================================
# LOAD WHISPERX MODEL
# ============================================================

print("Loading WhisperX model...")

MODEL = whisperx.load_model(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

print(f"WhisperX loaded successfully on {DEVICE}")


# ============================================================
# EXTRACT AUDIO FROM VIDEO
# ============================================================

def extract_audio(video_path, output_audio_path=None):
    """
    Extract audio from a video using FFmpeg.

    Parameters
    ----------
    video_path : str
        Path to the input video.

    output_audio_path : str, optional
        Path where the extracted WAV audio will be saved.

    Returns
    -------
    str
        Path to the extracted audio file.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    if output_audio_path is None:

        base_name = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        output_audio_path = os.path.join(
            os.path.dirname(video_path),
            f"{base_name}_audio.wav"
        )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,

        # Remove video stream
        "-vn",

        # Mono audio
        "-ac",
        "1",

        # 16 kHz sample rate
        "-ar",
        "16000",

        # Uncompressed WAV
        "-c:a",
        "pcm_s16le",

        output_audio_path
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    return output_audio_path


# ============================================================
# TRANSCRIBE AUDIO
# ============================================================

def transcribe_audio(audio_path):
    """
    Transcribe an audio file using WhisperX.

    Parameters
    ----------
    audio_path : str
        Path to WAV/audio file.

    Returns
    -------
    dict
        WhisperX transcription result.
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    audio = whisperx.load_audio(audio_path)

    result = MODEL.transcribe(
        audio,
        batch_size=BATCH_SIZE
    )

    return result


# ============================================================
# WORD-LEVEL ALIGNMENT
# ============================================================

def align_transcription(result, audio_path):
    """
    Perform word-level alignment using WhisperX.

    Parameters
    ----------
    result : dict
        Output returned by WhisperX transcription.

    audio_path : str
        Path to the audio file.

    Returns
    -------
    dict
        Word-aligned WhisperX result.
    """

    language = result["language"]

    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=DEVICE
    )

    audio = whisperx.load_audio(audio_path)

    aligned_result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False
    )

    return aligned_result


# ============================================================
# EXTRACT WORD TIMESTAMPS
# ============================================================

def extract_word_timestamps(aligned_result):
    """
    Extract clean word-level timestamps.

    Returns
    -------
    list
        List containing word, start time and end time.
    """

    words = []

    for segment in aligned_result["segments"]:

        if "words" not in segment:
            continue

        for word in segment["words"]:

            if "start" not in word:
                continue

            if "end" not in word:
                continue

            word_text = word["word"].strip()

            if not word_text:
                continue

            words.append({
                "word": word_text,
                "start": round(word["start"], 3),
                "end": round(word["end"], 3)
            })

    return words


# ============================================================
# SAVE JSON
# ============================================================

def save_json(data, output_path):
    """
    Save transcription information as JSON.
    """

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# SAVE TEXT TRANSCRIPT
# ============================================================

def save_transcript(text, output_path):
    """
    Save plain-text transcript.
    """

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(text)


# ============================================================
# MAIN VIDEO PROCESSING FUNCTION
# ============================================================

def process_video(video_path, output_directory="outputs"):
    """
    Complete audio-processing pipeline.

    Pipeline:

        Video
          ↓
        FFmpeg
          ↓
        Audio extraction
          ↓
        WhisperX transcription
          ↓
        Word-level alignment
          ↓
        Timestamp extraction
          ↓
        JSON + TXT output

    Parameters
    ----------
    video_path : str
        Path to input video.

    output_directory : str
        Directory where results will be saved.

    Returns
    -------
    dict
        Complete processing result.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Generate filenames
    # --------------------------------------------------------

    base_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    audio_path = os.path.join(
        output_directory,
        f"{base_name}_audio.wav"
    )

    json_path = os.path.join(
        output_directory,
        f"{base_name}_audio_word_timestamps.json"
    )

    text_path = os.path.join(
        output_directory,
        f"{base_name}_audio_transcript.txt"
    )

    # --------------------------------------------------------
    # 1. Extract audio
    # --------------------------------------------------------

    print("\n[1/5] Extracting audio...")

    extract_audio(
        video_path,
        audio_path
    )

    print("Audio extracted:", audio_path)

    # --------------------------------------------------------
    # 2. Transcription
    # --------------------------------------------------------

    print("\n[2/5] Transcribing audio...")

    result = transcribe_audio(
        audio_path
    )

    language = result["language"]

    print("Detected language:", language)

    # --------------------------------------------------------
    # 3. Word alignment
    # --------------------------------------------------------

    print("\n[3/5] Generating word-level timestamps...")

    aligned_result = align_transcription(
        result,
        audio_path
    )

    # --------------------------------------------------------
    # 4. Extract timestamps
    # --------------------------------------------------------

    print("\n[4/5] Extracting word timestamps...")

    words = extract_word_timestamps(
        aligned_result
    )

    # --------------------------------------------------------
    # Create complete result
    # --------------------------------------------------------

    transcript = " ".join(
        word["word"]
        for word in words
    )

    complete_result = {

        "video_file": video_path,

        "audio_file": audio_path,

        "language": language,

        "transcript": transcript,

        "words": words
    }

    # --------------------------------------------------------
    # 5. Save results
    # --------------------------------------------------------

    print("\n[5/5] Saving results...")

    save_json(
        complete_result,
        json_path
    )

    save_transcript(
        transcript,
        text_path
    )

    print("\nProcessing completed successfully!")

    print("JSON:", json_path)
    print("Transcript:", text_path)

    return complete_result


# ============================================================
# OPTIONAL DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    print("\nWhisperX Audio Module")
    print("---------------------")

    video_path = input(
        "Enter the path of the video file: "
    ).strip()

    result = process_video(
        video_path
    )

    print("\nTranscript:")
    print(result["transcript"])
