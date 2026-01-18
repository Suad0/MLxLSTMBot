#!/bin/bash

# MLXSTMBot Build Script
# This script builds the xLSTM project using Xcode

set -e  # Exit on any error

echo "🔨 Building MLXSTMBot..."
echo "=========================="

# Clean previous build (optional - uncomment if needed)
# echo "Cleaning previous build..."
# xcodebuild -project MLXSTMBot.xcodeproj -scheme MLXSTMBot clean

# Build the project
echo "Building project..."
xcodebuild -project MLXSTMBot.xcodeproj -scheme MLXSTMBot build -derivedDataPath build

# Find and copy the executable to a predictable location
BUILT_APP=$(find build -name "MLXSTMBot" -type f -path "*/Products/*" | head -1)
if [ -n "$BUILT_APP" ]; then
    cp "$BUILT_APP" ./build/MLXSTMBot
    echo "Copied executable to ./build/MLXSTMBot"
fi

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build completed successfully!"
    echo ""
    echo "Executable location:"
    find . -name "MLXSTMBot" -type f 2>/dev/null | head -1
    echo ""
    echo "To run training, use: ./train.sh"
else
    echo ""
    echo "❌ Build failed!"
    exit 1
fi