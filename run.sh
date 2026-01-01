#!/bin/bash

# MLXSTMBot Runner Script
# This script can build and run the xLSTM project in different modes

set -e  # Exit on any error

# Function to display usage
usage() {
    echo "Usage: $0 [build|train|test|interactive|clean]"
    echo ""
    echo "Commands:"
    echo "  build       - Build the project"
    echo "  train       - Run training (builds first if needed)"
    echo "  test        - Run in test mode"
    echo "  interactive - Run in interactive mode"
    echo "  clean       - Clean build artifacts"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 train"
    echo "  $0 test"
    exit 1
}

# Function to build the project
build_project() {
    echo "🔨 Building MLXSTMBot..."
    echo "=========================="
    
    xcodebuild -project MLXSTMBot.xcodeproj -scheme MLXSTMBot build
    
    if [ $? -eq 0 ]; then
        echo "✅ Build completed successfully!"
        return 0
    else
        echo "❌ Build failed!"
        return 1
    fi
}

# Function to find executable
find_executable() {
    find . -name "MLXSTMBot" -type f 2>/dev/null | head -1
}

# Function to run the executable
run_executable() {
    local mode=$1
    
    EXECUTABLE=$(find_executable)
    
    if [ -z "$EXECUTABLE" ]; then
        echo "❌ MLXSTMBot executable not found!"
        echo "Building project first..."
        build_project || exit 1
        EXECUTABLE=$(find_executable)
    fi
    
    if [ -z "$EXECUTABLE" ]; then
        echo "❌ Still no executable found after build!"
        exit 1
    fi
    
    echo "Found executable: $EXECUTABLE"
    
    # Make sure it's executable
    chmod +x "$EXECUTABLE"
    
    # Run with the specified mode
    echo "🚀 Running: $EXECUTABLE $mode"
    echo ""
    "$EXECUTABLE" "$mode"
}

# Function to clean build artifacts
clean_project() {
    echo "🧹 Cleaning build artifacts..."
    echo "=============================="
    
    xcodebuild -project MLXSTMBot.xcodeproj -scheme MLXSTMBot clean
    
    # Also remove any found executables
    EXECUTABLE=$(find_executable)
    if [ -n "$EXECUTABLE" ]; then
        echo "Removing executable: $EXECUTABLE"
        rm -f "$EXECUTABLE"
    fi
    
    echo "✅ Clean completed!"
}

# Main script logic
case "${1:-}" in
    "build")
        build_project
        ;;
    "train")
        echo "🚀 Starting xLSTM Training..."
        echo "============================="
        run_executable "train"
        ;;
    "test")
        echo "🧪 Running xLSTM Tests..."
        echo "========================="
        run_executable "test"
        ;;
    "interactive")
        echo "💬 Starting Interactive Mode..."
        echo "=============================="
        run_executable "interactive"
        ;;
    "clean")
        clean_project
        ;;
    *)
        usage
        ;;
esac