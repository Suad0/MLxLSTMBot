#!/bin/bash
xcodebuild -project MLXSTMBot.xcodeproj -scheme MLXSTMBot -configuration Debug CONFIGURATION_BUILD_DIR=$(pwd)/build && ./build/MLXSTMBot
