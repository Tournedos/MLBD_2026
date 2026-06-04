# Local configuration for the repository.
# Copy or edit this file to point to your data directory.
import os

# Default: repository-relative data/ folder
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Optional: directory to write figures / outputs
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

def get_data_dir():
    return os.environ.get("GOGYMI_DATA", DATA_DIR)
