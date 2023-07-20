#!/bin/bash

if [ ! -d "venv" ]; then
  # Create virtual environment
  python -m venv env
fi

# Activate virtual environment
source ./env/Scripts/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install python dependencies from requirements.txt
pip install -r requirements.txt
