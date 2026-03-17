"""
Automated model downloader for Netra.
Downloads buffalo_l from InsightFace and extracts the two ONNX files.
"""
import urllib.request
import zipfile
import shutil
from pathlib import Path


MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
MODEL_DIR = Path("models")

FILES_TO_EXTRACT = {
    "buffalo_l/det_10g.onnx":    "retinaface_10g.onnx",
    "buffalo_l/w600k_r50.onnx":  "arcface_w600k_r50.onnx",
}


def download_with_progress(url: str, dest: Path):
    print(f"Downloading {url}")
    print("This is ~300 MB — may take a few minutes...")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 / total_size
            mb_done = count * block_size / 1_000_000
            mb_total = total_size / 1_000_000
            print(f"\r  {pct:.1f}%  ({mb_done:.1f} / {mb_total:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print()  # newline after progress


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    zip_path = MODEL_DIR / "buffalo_l.zip"

    # Download
    if not zip_path.exists():
        download_with_progress(MODEL_URL, zip_path)
    else:
        print(f"Zip already exists at {zip_path}, skipping download.")

    # Extract only the two files we need
    print("Extracting ONNX models...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        available = zf.namelist()
        print(f"  Files in zip: {available}")

        for zip_name, dest_name in FILES_TO_EXTRACT.items():
            dest_path = MODEL_DIR / dest_name
            if dest_path.exists():
                print(f"  Already exists: {dest_name}")
                continue

            # Find the file (path inside zip may vary slightly)
            match = next((n for n in available if n.endswith(zip_name.split("/")[-1])), None)
            if match is None:
                print(f"  WARNING: {zip_name} not found in zip. Contents: {available}")
                continue

            print(f"  Extracting {match} → {dest_name}")
            with zf.open(match) as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    # Verify
    print("\nVerifying extracted models:")
    all_ok = True
    for dest_name in FILES_TO_EXTRACT.values():
        p = MODEL_DIR / dest_name
        if p.exists():
            size_mb = p.stat().st_size / 1_000_000
            print(f"  ✓  {dest_name}  ({size_mb:.1f} MB)")
        else:
            print(f"  ✗  MISSING: {dest_name}")
            all_ok = False

    if all_ok:
        print("\nModels ready. You can now run: python main.py")
        # Optionally remove the zip to save space
        answer = input("Delete the zip file to save ~300 MB? [y/N] ").strip().lower()
        if answer == "y":
            zip_path.unlink()
            print("Zip deleted.")
    else:
        print("\nSome models are missing. Check warnings above.")


if __name__ == "__main__":
    main()