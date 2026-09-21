from pathlib import Path

import qrcode


SITE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = SITE_ROOT / "assets" / "qrcodes"
PAGES = {
    "hwb-plus-audio-demo.png": "https://liuxin0.github.io/hwb-plus/",
    "switchse-audio-demo.png": "https://liuxin0.github.io/switchse/",
}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for filename, url in PAGES.items():
        code = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=24,
            border=4,
        )
        code.add_data(url)
        code.make(fit=True)
        image = code.make_image(fill_color="black", back_color="white")
        image.save(OUTPUT_ROOT / filename, format="PNG", optimize=True)
        print(f"Generated {filename}: {url}")


if __name__ == "__main__":
    main()
