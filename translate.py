import os
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")       
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"               
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"              
os.environ["ABSL_LOGGING_MIN_LOG_LEVEL"] = "3"         
os.environ["GLOG_minloglevel"] = "3"                   
os.environ["GRPC_VERBOSITY"] = "ERROR"                 
os.environ["GRPC_TRACE"] = ""                      
os.environ["PYTHONWARNINGS"] = "ignore"            
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  
os.environ.setdefault("AUDIODEV", "null")

os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"    
os.environ["TRANSFORMERS_VERBOSITY"] = "error"      
os.environ["TOKENIZERS_PARALLELISM"] = "false"      


import argparse
import subprocess
import srt
from datetime import timedelta

import numpy as np
import librosa
import soundfile as sf
import moviepy.editor as mpy
import torch
from deep_translator import GoogleTranslator
from TTS.api import TTS
import tempfile



def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--video_file", required=True)
    p.add_argument("--transcript_srt", required=True)
    p.add_argument("--output_dir", default="output")
    p.add_argument("--do_lipsync")
    p.add_argument("--min_gap", type=float, default=0.12)
    p.add_argument("--trim_db", type=float, default=35.0)
    return p.parse_args()


def translate_en_to_de(text: str) -> str:
    # Translate wrapper
    return GoogleTranslator(source="en", target="de").translate(text)


def trim_silence(y: np.ndarray, sr: int, top_db: float = 35.0) -> np.ndarray:
    if y.size == 0:
        return y
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed if y_trimmed.size > 0 else y


def synthesize_line(tts, text_de: str, speaker_wav: str, language: str = "de") -> tuple[np.ndarray, int]:
    """
    Returns (mono float32 waveform, sample_rate).
    """
    # Create temp wav path
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        # Write TTS directly to file
        tts.tts_to_file(text=text_de, speaker_wav=speaker_wav, language=language, file_path=tmp_path)
        y, sr = sf.read(tmp_path, dtype="float32")
        # Downmix to mono if needed
        if y.ndim == 2:
            y = y.mean(axis=1)
        return y.astype(np.float32), int(sr)
    except Exception:
        # Short silence at 24 kHz
        sr = 24000
        y = np.zeros(int(sr * 0.25), dtype=np.float32)
        return y, sr
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def ensure_length(y: np.ndarray, sr: int, min_seconds: float = 0.25) -> np.ndarray:
    """Pad short lines so they remain audible long enough."""
    min_len = int(sr * min_seconds)
    if y.size < min_len:
        pad = np.zeros(min_len - y.size, dtype=y.dtype)
        y = np.concatenate([y, pad])
    return y


def build_german_audio_and_timeline(subs, tts, speaker_wav: str, min_gap: float, trim_db: float):
    """
    For each subtitle:
      -Translate to german
      -TTS to get German speech audio and duration
    Return:
      -Concatenated German audio (np.ndarray, sr)
      -List of (start_sec, end_sec, text_de) before time-stretch
      -total_duration_sec
    """
    german_lines = []
    segments = []      
    segments_len = []  
    gaps = []          
    device_sr = None

    # Precompute original gaps
    for i in range(len(subs)):
        if i < len(subs) - 1:
            gap = (subs[i+1].start - subs[i].end).total_seconds()
            gaps.append(max(min_gap, max(0.0, gap)))
        else:
            gaps.append(0.0)  # no gap after last line

    # Translate and synthesize each line
    for i, sub in enumerate(subs):
        en = sub.content.strip()
        de = translate_en_to_de(en) if en else ""
        audio, sr = synthesize_line(tts, de, speaker_wav, language="de")
        device_sr = device_sr or sr
        if device_sr is None:
            device_sr = sr
        elif sr != device_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=device_sr)
            sr = device_sr
        audio = trim_silence(audio, sr, top_db=trim_db)
        audio = ensure_length(audio, sr, min_seconds=0.25)

        dur = librosa.get_duration(y=audio, sr=sr)
        segments.append(audio)
        segments_len.append(dur)
        german_lines.append(de)

    # Concatenate with original gaps
    sr = device_sr or 22050
    concat_audio = []
    timeline = []  # (start_sec, end_sec, text_de) before stretch
    cursor = 0.0

    for i, (seg, seg_len, text_de) in enumerate(zip(segments, segments_len, german_lines)):
        # Add speech
        concat_audio.append(seg)
        start = cursor
        end = cursor + seg_len
        timeline.append((start, end, text_de))
        cursor = end

        # Add gap the line
        gap_sec = gaps[i]
        if gap_sec > 0:
            silence = np.zeros(int(round(sr * gap_sec)), dtype=np.float32)
            concat_audio.append(silence)
            cursor += gap_sec

    y_full = np.concatenate(concat_audio) if len(concat_audio) else np.zeros(0, dtype=np.float32)
    total_dur = librosa.get_duration(y=y_full, sr=sr) if y_full.size > 0 else 0.0
    return y_full, sr, timeline, total_dur


def scale_timeline(timeline, rate: float):
    scaled = []
    inv = 1.0 / rate if rate != 0 else 1.0
    for start, end, text in timeline:
        scaled.append((start * inv, end * inv, text))
    return scaled


def write_srt_from_timeline(timeline_scaled, out_path: str):
    items = []
    for idx, (start_s, end_s, text) in enumerate(timeline_scaled, start=1):
        # Ensure non-negative and minimum 0.3s visibility
        start_s = max(0.0, start_s)
        if end_s <= start_s + 0.3:
            end_s = start_s + 0.3
        items.append(
            srt.Subtitle(
                index=idx,
                start=timedelta(seconds=start_s),
                end=timedelta(seconds=end_s),
                content=text or ""
            )
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(items))


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load video and audio
    video = mpy.VideoFileClip(args.video_file)
    original_duration = video.duration
    original_audio_path = os.path.join(args.output_dir, "original_audio.wav")
    video.audio.write_audiofile(original_audio_path, verbose=False, logger=None)

    # Read English SRT
    with open(args.transcript_srt, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # German audio and timeline
    y_de, sr, timeline, dur_de = build_german_audio_and_timeline(
        subs=subs,
        tts=tts,
        speaker_wav=original_audio_path,
        min_gap=args.min_gap,
        trim_db=args.trim_db
    )

    translated_audio_path_raw = os.path.join(args.output_dir, "translated_audio_raw.wav")
    sf.write(translated_audio_path_raw, y_de, sr)
    print(f"Raw German audio: {translated_audio_path_raw} (duration: {dur_de:.2f}s)")

    # Uniformly adjust audio speed to match original video duration
    if dur_de > 0 and original_duration > 0:
        rate = dur_de / float(original_duration)
    else:
        rate = 1.0

    if rate != 1.0:
        y_adj = librosa.effects.time_stretch(y_de, rate=rate)
    else:
        y_adj = y_de

    adjusted_audio_path = os.path.join(args.output_dir, "translated_audio.wav")
    sf.write(adjusted_audio_path, y_adj, sr)
    print(f"Adjusted German audio: {adjusted_audio_path}")

    # Scale German timeline so SRT matches altered audio
    timeline_scaled = scale_timeline(timeline, rate=rate)

    # Write German SRT
    translated_srt_path = os.path.join(args.output_dir, "translated.srt")
    write_srt_from_timeline(timeline_scaled, translated_srt_path)
    print(f"German-aligned SRT written: {translated_srt_path}")

    # Final video with the adjusted German audio
    new_audio = mpy.AudioFileClip(adjusted_audio_path)
    new_video = video.set_audio(new_audio)
    translated_video_path = os.path.join(args.output_dir, "translated_video.mp4")
    new_video.write_videofile(
        translated_video_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=os.path.join(args.output_dir, "temp-audio.m4a"),
        remove_temp=True,
        verbose=False,
        logger=None
    )
    print(f"Final translated video: {translated_video_path}")

    # Wav2Lip lipsync
    if args.do_lipsync:
        wav2lip_dir = "Wav2Lip"
        checkpoint_path = os.path.join(wav2lip_dir, "checkpoints", "wav2lip_gan.pth")
        lipsync_video_path = os.path.join(args.output_dir, "lipsync_video.mp4")
        cmd = [
            "python", os.path.join(wav2lip_dir, "inference.py"),
            "--checkpoint_path", checkpoint_path,
            "--face", translated_video_path,
            "--audio", adjusted_audio_path,
            "--outfile", lipsync_video_path
        ]
        subprocess.run(cmd, check=False)
        print(f"Lip-synced video: {lipsync_video_path}")


if __name__ == "__main__":
    main()
