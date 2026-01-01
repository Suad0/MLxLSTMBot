#!/bin/bash

# MLXSTMBot Training Script
# This script runs the xLSTM training process

set -e  # Exit on any error

echo "🚀 Starting xLSTM Training..."
echo "============================="

# Find the built executable
EXECUTABLE=$(find . -name "MLXSTMBot" -type f 2>/dev/null | head -1)

if [ -z "$EXECUTABLE" ]; then
    echo "❌ MLXSTMBot executable not found!"
    echo "Please run ./build.sh first to build the project."
    exit 1
fi

echo "Found executable: $EXECUTABLE"
echo ""

# Check if executable exists and is executable
if [ ! -f "$EXECUTABLE" ]; then
    echo "❌ Executable file not found: $EXECUTABLE"
    exit 1
fi

if [ ! -x "$EXECUTABLE" ]; then
    echo "Making executable..."
    chmod +x "$EXECUTABLE"
fi

# Run the training
echo "Starting training with command: $EXECUTABLE train"
echo "Press Ctrl+C to stop training"
echo ""

# Execute the training
"$EXECUTABLE" train

echo ""
echo "Training completed!"