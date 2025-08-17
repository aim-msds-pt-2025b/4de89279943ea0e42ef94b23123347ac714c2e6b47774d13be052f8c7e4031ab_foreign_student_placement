"""Simple data ingestion utility for the ML pipeline."""

from pathlib import Path
import shutil


def load_dataset(src_path: str = "data/global_student_migration.csv", dest_dir: str = "data") -> str:
    """Ensure dataset is present under dest_dir and return its path.

    If src_path is elsewhere, copy it into dest_dir preserving filename.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    dest_path = dest / Path(src_path).name
    src = Path(src_path)
    if src.resolve() != dest_path.resolve() and src.exists():
        shutil.copy2(src, dest_path)
    return str(dest_path)
