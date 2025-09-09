#!/bin/bash

# Download GIGA model checkpoint from Google Drive
# Usage: bash download_checkpoint.sh [mvh|dna]

set -e

source .venv/bin/activate

# Check for dataset argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 [mvh|dna]"
    echo "  mvh - Download MVHumanNet checkpoint"
    echo "  dna - Download DNA-Rendering checkpoint"
    exit 1
fi

DATASET="$1"

# Set file ID and output filename based on dataset
case "$DATASET" in
    mvh)
        FILE_ID="1Qva5M7nk9Cgu2SDdgdUa77P98tjz560p"
        OUTPUT_FILE="./configs/mvh/mvh.ckpt"
        echo "Downloading MVHumanNet checkpoint..."
        ;;
    dna)
        FILE_ID="1LA1rrA6um2cnarHEtC4aPPKvJQuTxy4E"
        OUTPUT_FILE="./configs/dna/dna.ckpt"
        echo "Downloading DNA-Rendering checkpoint..."
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET'"
        echo "Use 'mvh' or 'dna'"
        exit 1
        ;;
esac

echo "Downloading checkpoint from Google Drive..."
gdown "$FILE_ID" --output "$OUTPUT_FILE"

echo "Download completed: $OUTPUT_FILE"
echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"

deactivate

exit 0
