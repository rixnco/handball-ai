#!/usr/bin/env bash
# Quick local setup for handball-ai
# Usage: bash scripts/setup_local.sh

set -e

echo "=== handball-ai: local setup ==="

# Check Python 3.11+
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python found: $PYTHON_VERSION"

# Create the venv when missing
if [ ! -d ".venv" ]; then
    echo "Creating the venv .venv/"
    python3 -m venv .venv
fi

# Activate the venv and install
echo "Activating the venv and installing the dependencies"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Check the install
echo ""
echo "=== Check ==="
python -c "import handball_ai; print(f'handball_ai version {handball_ai.__version__}')"
python -c "from handball_ai.config import CourtConfig; c = CourtConfig(); print(f'Court: {c.length_m}x{c.width_m} m')"

echo ""
echo "=== Running the tests ==="
pytest tests/ -v --tb=short

echo ""
echo "=== Setup OK ==="
echo "To activate the venv later: source .venv/bin/activate"
