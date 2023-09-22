#!/bin/bash

if ! command -v pip &> /dev/null; then
    echo "pip is not installed. Please install pip first."
    exit 1
fi

if [[ -f "requirements.txt" ]]; then
    python -m pip install --upgrade pip

    pip install -r requirements.txt
else
    echo "requirements.txt not found in the current directory."
    exit 1
fi
