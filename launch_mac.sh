#!/bin/bash
# Launch Chatterbox Audiobook on macOS

ENV_NAME="chatterbox"
PROJECT_DIR="/Volumes/Exp/AI_Work/chatterbox-audiobook"

# Check if the chatterbox conda environment exists
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "ERROR: conda environment '${ENV_NAME}' not found."
    echo ""
    echo "To create it, run the following commands:"
    echo ""
    echo "  conda create -n chatterbox python=3.10"
    echo "  conda activate chatterbox"
    echo "  conda install pytorch::pytorch torchvision torchaudio -c pytorch"
    echo "  pip install \"setuptools<70\""
    echo "  pip install -r requirements-mac.txt"
    echo "  pip install -e . --no-deps"
    echo "  python -m spacy download en_core_web_sm"
    echo ""
    exit 1
fi

echo "Launching Chatterbox Audiobook..."
# PYTORCH_ENABLE_MPS_FALLBACK lets MPS use CPU only for ops it can't handle
# (e.g. conv layers with >65536 channels). Everything else stays on Metal GPU.
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n "${ENV_NAME}" --no-capture-output \
    python "${PROJECT_DIR}/gradio_audiobook_app.py"
