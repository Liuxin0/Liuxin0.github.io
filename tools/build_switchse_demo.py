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
ASSET_VERSION = "switchse-demo-v2-20260921"
TARGET_PEAK = 10 ** (-1.0 / 20.0)
SPECTROGRAM_GAIN_DB = 20.0
SPECTROGRAM_RANGE_DB = 80.0


DATASETS = {
    "CHiME": {
        "group": "chime",
        "collection": "CHiME-3 target-domain test set",
        "enhancement_dir": "CHiME-Enh",
        "asr_dir": "CHiME-ASR",
        "variants": (("Noisy input", "Real noisy recording", "enhancement", 0, "noisy"), ("S_Enh", "Speech-quality-oriented mode", "enhancement", 1, "s-enh"), ("S_ASR", "ASR-oriented mode", "asr", 1, "s-asr")),
    },
    "VCTK": {
        "group": "vctk",
        "collection": "VCTK source-domain test set",
        "enhancement_dir": "VCTK-Enh",
        "asr_dir": "VCTK-ASR",
        "variants": (("Noisy input", "Synthetic noisy mixture", "enhancement", 0, "noisy"), ("GCRN", "Pre-trained baseline", "enhancement", 1, "gcrn"), ("S_Enh", "Speech-quality-oriented mode", "enhancement", 2, "s-enh"), ("S_ASR", "ASR-oriented mode", "asr", 2, "s-asr")),
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


def comparison_key(stem: str) -> str:
    base, separator, mode = stem.rpartition("_")
    if not separator or mode not in {"0.0", "1.0"}:
        raise ValueError(f"Expected a mode-tagged source file name: {stem}")
    return base


def source_index(directory: Path) -> dict[str, Path]:
    indexed = {}
    for path in sorted(directory.glob("*.wav")):
        key = comparison_key(path.stem)
        if key in indexed:
            raise ValueError(f"Duplicate utterance key {key} in {directory}")
        indexed[key] = path
    return indexed


def validate_shared_channels(dataset_name: str, utterance_key: str, enhancement: np.ndarray, asr: np.ndarray) -> None:
    if enhancement.shape[0] != asr.shape[0]:
        raise ValueError(f"Mismatched duration for {dataset_name} {utterance_key}")
    if not np.array_equal(enhancement[:, 0], asr[:, 0]):
        raise ValueError(f"Noisy input differs between modes for {dataset_name} {utterance_key}")


def title_for(dataset_name: str, key: str) -> str:
    parts = key.split("_")
    if dataset_name == "CHiME":
        return f"{parts[0]} · {parts[-2]} · real noisy speech"
    return f"{parts[0].upper()} / {parts[1]} · {parts[2]} dB mixture"


def asset_url(path: Path) -> str:
    relative = path.relative_to(DEMO_ROOT).as_posix()
    return f"{relative}?v={ASSET_VERSION}"


def main() -> None:
    samples = []
    for dataset_name, config in DATASETS.items():
        enhancement_paths = source_index(SOURCE_ROOT / config["enhancement_dir"])
        asr_paths = source_index(SOURCE_ROOT / config["asr_dir"])
        if enhancement_paths.keys() != asr_paths.keys():
            missing_asr = sorted(enhancement_paths.keys() - asr_paths.keys())
            missing_enhancement = sorted(asr_paths.keys() - enhancement_paths.keys())
            raise ValueError(f"Unpaired {dataset_name} files; missing ASR={missing_asr}, missing enhancement={missing_enhancement}")

        for key in sorted(enhancement_paths):
            enhancement, enhancement_rate = read_wav(enhancement_paths[key])
            asr, asr_rate = read_wav(asr_paths[key])
            expected_channels = 2 if dataset_name == "CHiME" else 3
            if enhancement_rate != asr_rate or enhancement.shape[1] != expected_channels or asr.shape[1] != expected_channels:
                raise ValueError(f"Unexpected audio format for {dataset_name} {key}")
            validate_shared_channels(dataset_name, key, enhancement, asr)

            sources = {"enhancement": enhancement, "asr": asr}
            selected = [(label, detail, sources[mode][:, channel], suffix) for label, detail, mode, channel, suffix in config["variants"]]
            peak = float(np.max(np.abs(np.stack([channel for _, _, channel, _ in selected], axis=1))))
            gain = TARGET_PEAK / peak if peak > 0 else 1.0
            variants = []
            for label, detail, channel, suffix in selected:
                normalized = channel * gain
                audio_path = AUDIO_ROOT / config["group"] / f"{key}-{suffix}.wav"
                image_path = SPECTROGRAM_ROOT / config["group"] / f"{key}-{suffix}.png"
                write_mono_wav(normalized, enhancement_rate, audio_path)
                save_spectrogram(normalized, enhancement_rate, image_path)
                variants.append({"label": label, "detail": detail, "audio": asset_url(audio_path), "spectrogram": asset_url(image_path)})

            samples.append({"id": f"{config['group']}-{key}", "group": config["group"], "collection": config["collection"], "title": title_for(dataset_name, key), "sampleRate": enhancement_rate, "duration": round(len(enhancement) / enhancement_rate, 3), "variants": variants})

    MANIFEST_PATH.write_text(json.dumps({"samples": samples}, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(samples)} SwitchSE audio comparisons.")


if __name__ == "__main__":
    main()
