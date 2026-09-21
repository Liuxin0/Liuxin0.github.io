from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import numpy as np
from PIL import Image


SITE_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = SITE_ROOT / "hwb-plus"
AUDIO_ROOT = DEMO_ROOT / "audio"
SPECTROGRAM_ROOT = DEMO_ROOT / "spectrograms"
MANIFEST_PATH = DEMO_ROOT / "comparisons.json"
ASSET_VERSION = "hwb-fixed-level-spectrogram-20260921"


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    payload = path.read_bytes()
    if payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError(f"Unsupported WAV container: {path.name}")

    offset = 12
    format_chunk = data_chunk = None
    while offset + 8 <= len(payload):
        chunk_id = payload[offset:offset + 4]
        chunk_size = struct.unpack_from("<I", payload, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_id == b"fmt ":
            format_chunk = payload[chunk_start:chunk_end]
        elif chunk_id == b"data":
            data_chunk = payload[chunk_start:chunk_end]
        offset = chunk_end + chunk_size % 2

    if format_chunk is None or data_chunk is None or len(format_chunk) < 16:
        raise ValueError(f"Incomplete WAV data: {path.name}")

    format_tag, channel_count, sample_rate, _, _, bits = struct.unpack_from("<HHIIHH", format_chunk)
    if format_tag == 0xFFFE and len(format_chunk) >= 40:
        format_tag = struct.unpack_from("<H", format_chunk, 24)[0]

    if format_tag == 1 and bits == 16:
        samples = np.frombuffer(data_chunk, dtype="<i2").astype(np.float32) / 32768.0
    elif format_tag == 1 and bits == 32:
        samples = np.frombuffer(data_chunk, dtype="<i4").astype(np.float32) / 2147483648.0
    elif format_tag == 3 and bits == 32:
        samples = np.frombuffer(data_chunk, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV encoding in {path.name}: format={format_tag}, bits={bits}")

    if len(samples) % channel_count:
        raise ValueError(f"Invalid channel alignment: {path.name}")
    return samples.reshape(-1, channel_count), sample_rate


def write_mono_wav(samples: np.ndarray, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0 - 1 / 32768).astype(np.float32)
    pcm = (pcm * 32768.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def colorize(values: np.ndarray) -> np.ndarray:
    stops = np.array(
        [[0.00, 14, 25, 29], [0.28, 22, 66, 70], [0.52, 43, 143, 136], [0.74, 234, 169, 80], [1.00, 255, 242, 194]],
        dtype=np.float32,
    )
    return np.stack([np.interp(values, stops[:, 0], stops[:, index]) for index in range(1, 4)], axis=-1).astype(np.uint8)


def save_spectrogram(samples: np.ndarray, sample_rate: int, path: Path) -> None:
    frame_length, hop_length = 1024, 160
    if len(samples) < frame_length:
        samples = np.pad(samples, (0, frame_length - len(samples)))
    remainder = (len(samples) - frame_length) % hop_length
    if remainder:
        samples = np.pad(samples, (0, hop_length - remainder))
    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_length)[::hop_length]
    window = np.hanning(frame_length)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) / (window.sum() / 2.0)
    db = 20.0 * np.log10(np.maximum(spectrum, 1e-7))
    db = np.clip(db, -80.0, 0.0)
    maximum_bin = min(db.shape[1], int(11000 / sample_rate * frame_length) + 1)
    image = Image.fromarray(colorize(((db + 80.0) / 80.0)[:, :maximum_bin].T[::-1, :]), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((1200, 336), Image.Resampling.BICUBIC).save(path, format="PNG", optimize=True)


def a_weighted_rms(samples: np.ndarray, sample_rate: int) -> float:
    spectrum = np.fft.rfft(samples)
    frequencies = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    squared = frequencies**2
    numerator = (12200**2) * (frequencies**4)
    denominator = (
        (squared + 20.6**2)
        * np.sqrt((squared + 107.7**2) * (squared + 737.9**2))
        * (squared + 12200**2)
    )
    amplitude = np.zeros_like(frequencies)
    valid = denominator > 0
    amplitude[valid] = numerator[valid] / denominator[valid]
    gain = 10 ** ((20 * np.log10(np.maximum(amplitude, 1e-30)) + 2.0) / 20)
    energy = np.abs(spectrum) ** 2 * gain**2
    if len(samples) % 2 == 0:
        energy[1:-1] *= 2
    else:
        energy[1:] *= 2
    return float(np.sqrt(energy.sum() / len(samples) ** 2))


def label_from_identifier(identifier: str) -> tuple[str, str, str]:
    speaker, utterance, microphone = identifier.split("_")
    return speaker.upper(), utterance, microphone.upper()


def asset_url(path: Path) -> str:
    relative_path = str(path.relative_to(DEMO_ROOT)).replace("\\", "/")
    return f"{relative_path}?v={ASSET_VERSION}"


def main() -> None:
    source_files = sorted(
        path for path in AUDIO_ROOT.glob("p*_mic*.wav") if "_lr_" not in path.stem
    )
    comparisons = []
    channel_variants = (("lr", "LR input", "4 kHz bandwidth", 0), ("hwb-plus", "HWB-Plus", "Proposed", 1), ("hr", "HR reference", "Reference", 2))

    for source in source_files:
        identifier = source.stem
        multichannel, sample_rate = read_wav(source)
        if multichannel.shape[1] != 3:
            raise ValueError(f"Expected LR/predict/HR channels in {source.name}, got {multichannel.shape[1]}")

        variants = []
        for directory, label, detail, channel in channel_variants:
            audio_path = AUDIO_ROOT / directory / f"{identifier}.wav"
            spec_path = SPECTROGRAM_ROOT / directory / f"{identifier}.png"
            write_mono_wav(multichannel[:, channel], sample_rate, audio_path)
            save_spectrogram(multichannel[:, channel], sample_rate, spec_path)
            variants.append({
                "key": directory,
                "label": label,
                "detail": detail,
                "sampleRate": sample_rate,
                "duration": round(len(multichannel) / sample_rate, 2),
                "audio": asset_url(audio_path),
                "spectrogram": asset_url(spec_path),
            })

        source_hwb = AUDIO_ROOT / "hwb" / f"{identifier}_pr.wav"
        if not source_hwb.exists():
            raise FileNotFoundError(f"Missing HWB prediction: {source_hwb}")
        hwb_audio, hwb_rate = read_wav(source_hwb)
        if hwb_audio.shape[1] != 1:
            raise ValueError(f"Expected mono HWB prediction: {source_hwb.name}")
        hwb_target = AUDIO_ROOT / "hwb" / f"{identifier}.wav"
        hwb_signal = hwb_audio[:, 0]
        hwb_loudness = a_weighted_rms(hwb_signal, hwb_rate)
        hr_loudness = a_weighted_rms(multichannel[:, 2], sample_rate)
        if hwb_loudness <= 0 or hr_loudness <= 0:
            raise ValueError(f"Cannot calibrate silence for {identifier}")
        hwb_playback = hwb_signal * (hr_loudness / hwb_loudness)
        write_mono_wav(hwb_playback, hwb_rate, hwb_target)
        hwb_spec = SPECTROGRAM_ROOT / "hwb" / f"{identifier}.png"
        save_spectrogram(hwb_playback, hwb_rate, hwb_spec)
        variants.insert(1, {
            "key": "hwb",
            "label": "HWB",
            "detail": "Baseline",
            "sampleRate": hwb_rate,
            "duration": round(len(hwb_audio) / hwb_rate, 2),
            "audio": asset_url(hwb_target),
            "spectrogram": asset_url(hwb_spec),
        })

        speaker, utterance, microphone = label_from_identifier(identifier)
        comparisons.append({"id": identifier, "speaker": speaker, "utterance": utterance, "microphone": microphone, "variants": variants})

    MANIFEST_PATH.write_text(json.dumps({"comparisons": comparisons}, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(comparisons)} four-way comparisons.")


if __name__ == "__main__":
    main()
