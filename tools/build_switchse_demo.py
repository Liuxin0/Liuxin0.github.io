import json
import wave
from pathlib import Path

import numpy as np
from PIL import Image


SITE_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = SITE_ROOT / "switchse"
AUDIO_ROOT = DEMO_ROOT / "audio"
SPECTROGRAM_ROOT = DEMO_ROOT / "spectrograms"
MANIFEST_PATH = DEMO_ROOT / "samples.json"
SOURCE_ROOT = Path(r"D:\BaiduSyncdisk\documents\my paper\text_annotated-interspeech\展示\音频")
ASSET_VERSION = "switchse-demo-v1-20260921"
TARGET_PEAK = 10 ** (-1.0 / 20.0)
SPECTROGRAM_GAIN_DB = 20.0
SPECTROGRAM_RANGE_DB = 80.0


GROUPS = {
    "CHiME-Enh": {
        "group": "chime-enh",
        "collection": "CHiME-3 target domain · SEnh",
        "variant_labels": (("Noisy input", "Real noisy recording"), ("SwitchSE", "SEnh: quality-oriented mode")),
    },
    "CHiME-ASR": {
        "group": "chime-asr",
        "collection": "CHiME-3 target domain · SASR",
        "variant_labels": (("Noisy input", "Real noisy recording"), ("SwitchSE", "SASR: ASR-oriented mode")),
    },
    "VCTK-Enh": {
        "group": "vctk-enh",
        "collection": "VCTK source domain · SEnh",
        "variant_labels": (("Noisy input", "Synthetic noisy mixture"), ("GCRN", "Pre-trained baseline"), ("SwitchSE", "SEnh: quality-oriented mode")),
    },
    "VCTK-ASR": {
        "group": "vctk-asr",
        "collection": "VCTK source domain · SASR",
        "variant_labels": (("Noisy input", "Synthetic noisy mixture"), ("GCRN", "Pre-trained baseline"), ("SwitchSE", "SASR: ASR-oriented mode")),
    },
}


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as input_file:
        channels = input_file.getnchannels()
        sample_rate = input_file.getframerate()
        sample_width = input_file.getsampwidth()
        frames = input_file.getnframes()
        payload = input_file.readframes(frames)
    if sample_width != 2:
        raise ValueError(f"Expected PCM16 WAV input: {path}")
    samples = np.frombuffer(payload, dtype="<i2").reshape(-1, channels).astype(np.float32) / 32768.0
    return samples, sample_rate


def write_mono_wav(samples: np.ndarray, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0 - 1 / 32768.0)
    pcm = (pcm * 32768.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def colorize(values: np.ndarray) -> np.ndarray:
    stops = np.array(
        [
            [0.000, 0, 0, 0],
            [0.125, 0, 30, 60],
            [0.250, 56, 36, 136],
            [0.375, 132, 24, 160],
            [0.500, 196, 40, 132],
            [0.625, 236, 88, 88],
            [0.750, 252, 148, 72],
            [0.875, 240, 212, 148],
            [1.000, 255, 252, 250],
        ],
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
    values = np.clip((db + SPECTROGRAM_GAIN_DB + SPECTROGRAM_RANGE_DB) / SPECTROGRAM_RANGE_DB, 0.0, 1.0)
    maximum_bin = min(values.shape[1], int(8000 / sample_rate * frame_length) + 1)
    image = Image.fromarray(colorize(values[:, :maximum_bin].T[::-1, :]), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((1200, 336), Image.Resampling.BICUBIC).save(path, format="PNG", optimize=True)


def title_for(group_name: str, stem: str) -> str:
    parts = stem.split("_")
    if group_name.startswith("CHiME"):
        return f"{parts[0]} · {parts[-3]} · real noisy speech"
    return f"{parts[0].upper()} / {parts[1]} · {parts[2]} dB mixture"


def asset_url(path: Path) -> str:
    relative = path.relative_to(DEMO_ROOT).as_posix()
    return f"{relative}?v={ASSET_VERSION}"


def main() -> None:
    samples = []
    for group_name, config in GROUPS.items():
        source_dir = SOURCE_ROOT / group_name
        for source_path in sorted(source_dir.glob("*.wav")):
            multichannel, sample_rate = read_wav(source_path)
            labels = config["variant_labels"]
            if multichannel.shape[1] != len(labels):
                raise ValueError(f"Unexpected channel count in {source_path}: {multichannel.shape[1]}")

            peak = float(np.max(np.abs(multichannel)))
            gain = TARGET_PEAK / peak if peak > 0 else 1.0
            normalized = multichannel * gain
            identifier = f"{config['group']}-{source_path.stem}"
            variants = []
            for index, (label, detail) in enumerate(labels):
                audio_path = AUDIO_ROOT / config["group"] / f"{source_path.stem}-ch{index + 1}.wav"
                image_path = SPECTROGRAM_ROOT / config["group"] / f"{source_path.stem}-ch{index + 1}.png"
                write_mono_wav(normalized[:, index], sample_rate, audio_path)
                save_spectrogram(normalized[:, index], sample_rate, image_path)
                variants.append({"label": label, "detail": detail, "audio": asset_url(audio_path), "spectrogram": asset_url(image_path)})

            samples.append(
                {
                    "id": identifier,
                    "group": config["group"],
                    "collection": config["collection"],
                    "title": title_for(group_name, source_path.stem),
                    "sampleRate": sample_rate,
                    "duration": round(len(multichannel) / sample_rate, 3),
                    "variants": variants,
                }
            )

    MANIFEST_PATH.write_text(json.dumps({"samples": samples}, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(samples)} SwitchSE audio comparisons.")


if __name__ == "__main__":
    main()
