#!/usr/bin/env bash
# Deploy hf_space/ to Hugging Face Space s1m31/march-madness-2026
set -euo pipefail

REPO_ID="s1m31/march-madness-2026"
HF_CMD="/opt/anaconda3/bin/hf"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> Checking auth"
"$HF_CMD" auth whoami || {
    echo "Not logged in. Run:  hf auth login"
    exit 1
}

echo "==> Creating Space (no-op if already exists)"
"$HF_CMD" repo create "$REPO_ID" \
    --repo-type space \
    --space-sdk docker \
    --exist-ok

echo "==> Uploading folder"
"$HF_CMD" upload "$REPO_ID" "$HERE" . \
    --repo-type space \
    --commit-message "Deploy March Madness 2026 retrospective" \
    --exclude "**/__pycache__/**" \
    --exclude "deploy.sh" \
    --exclude "prepare_data.py" \
    --exclude ".DS_Store"

echo ""
echo "Done. Space should be building at:"
echo "  https://huggingface.co/spaces/$REPO_ID"
