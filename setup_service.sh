#!/bin/bash
# Setup AI Dataset API as a persistent system service (macOS)
# Located in project root for easy access

echo "🚀 Setting up AI Dataset API as a persistent service..."
echo "📍 Current directory: $(pwd)"

# Create LaunchAgent directory if it doesn't exist
mkdir -p ~/Library/LaunchAgents

# Create service definition
cat > ~/Library/LaunchAgents/com.ai.dataset.api.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ai.dataset.api</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(which python3)</string>
        <string>$(pwd)/deploy.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$(pwd)</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$(pwd)/logs/service.log</string>
    <key>StandardErrorPath</key>
    <string>$(pwd)/logs/service_error.log</string>
</dict>
</plist>
EOF

echo "✅ Service definition created"

# Load the service
launchctl load ~/Library/LaunchAgents/com.ai.dataset.api.plist

echo "🎉 AI Dataset API is now running as a persistent service!"
echo "📍 API available at: http://localhost:8000"
echo "📋 Logs: $(pwd)/logs/service.log"
echo ""
echo "🔧 Service Management Commands:"
echo "  Start:   launchctl load ~/Library/LaunchAgents/com.ai.dataset.api.plist"
echo "  Stop:    launchctl unload ~/Library/LaunchAgents/com.ai.dataset.api.plist"
echo "  Status:  launchctl list | grep com.ai.dataset.api"
echo ""
echo "🗑️  To remove service completely:"
echo "  launchctl unload ~/Library/LaunchAgents/com.ai.dataset.api.plist"
echo "  rm ~/Library/LaunchAgents/com.ai.dataset.api.plist"