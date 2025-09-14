import os
import shutil
import zipfile
import sys
import fnmatch
import tempfile

# ----------------------
# CONFIGURATION
# ----------------------
# Command-line season argument, defaults to NNLSeason8
SEASON = sys.argv[1] if len(sys.argv) > 1 else "NNLSeason8"

# Determine repo root (folder where this script lives)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Output zip path
ZIP_NAME = os.path.join(REPO_ROOT, f"MSB-League-Tools-{SEASON}.zip")

# Files and folders to exclude
EXCLUDE_FOLDERS = ["ExcelOut", "LeagueData", ".git"]
EXCLUDE_FILES = ["__pycache__", ".DS_Store", "PJC2_graphics_generator.py"]

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def should_exclude(path):
    """Return True if a file/folder should be excluded."""
    for folder in EXCLUDE_FOLDERS:
        if path.startswith(folder):
            return True
    for file in EXCLUDE_FILES:
        if fnmatch.fnmatch(os.path.basename(path), file):
            return True
    return False

# ----------------------
# CREATE TEMP STAGING FOLDER
# ----------------------
STAGING_DIR = tempfile.mkdtemp(prefix="msb_release_")

# ----------------------
# COPY FILES TO STAGING
# ----------------------
for root, dirs, files in os.walk(REPO_ROOT):
    # Skip the staging folder itself in case script is re-run inside it
    if os.path.abspath(root) == os.path.abspath(STAGING_DIR):
        continue

    rel_root = os.path.relpath(root, REPO_ROOT)

    # Skip excluded folders
    if should_exclude(rel_root) and rel_root != ".":
        dirs[:] = []  # prevent walking subdirs
        continue

    dest_root = os.path.join(STAGING_DIR, rel_root)
    os.makedirs(dest_root, exist_ok=True)

    for file in files:
        src_file = os.path.join(root, file)
        if should_exclude(file):
            continue
        dest_file = os.path.join(dest_root, file)
        shutil.copy2(src_file, dest_file)

# ----------------------
# COPY ONLY THE SPECIFIC SEASON
season_src = os.path.join(REPO_ROOT, "LeagueData", SEASON)
season_dest = os.path.join(STAGING_DIR, "LeagueData", SEASON)
os.makedirs(os.path.dirname(season_dest), exist_ok=True)
shutil.copytree(season_src, season_dest)

# CREATE ZIP FROM STAGING CONTENTS
with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(STAGING_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            # Store path relative to staging folder (not including staging folder itself)
            arcname = os.path.relpath(file_path, STAGING_DIR)
            zipf.write(file_path, arcname)


# CLEAN UP TEMP STAGING
shutil.rmtree(STAGING_DIR)

print(f"Created zip: {ZIP_NAME}")