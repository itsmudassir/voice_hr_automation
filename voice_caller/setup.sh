#!/bin/bash

# Twilio Voice Application Setup Script

echo "🚀 Setting up Twilio Voice Application"
echo "======================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env file..."
    cat > .env << EOL
# Twilio Configuration
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890

# Application Configuration
PORT=5000
DEBUG=True

# GitHub MCP Configuration (if using)
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_pat_here
EOL
    echo "✅ .env file created. Please update it with your Twilio credentials."
else
    echo ""
    echo "✅ .env file already exists."
fi

# Create ngrok configuration
echo ""
echo "📱 Setting up ngrok for local testing..."
echo "To test locally with Twilio, you'll need ngrok:"
echo "1. Download ngrok from https://ngrok.com/download"
echo "2. Run: ngrok http 5000"
echo "3. Use the HTTPS URL for your Twilio webhooks"

echo ""
echo "✨ Setup complete!"
echo ""
echo "📖 Next steps:"
echo "1. Update .env with your Twilio credentials"
echo "2. Run the inbound app: python app.py"
echo "3. Run the outbound app: python outbound_app.py"
echo "4. Set up ngrok for local testing"
echo "5. Configure Twilio webhooks:"
echo "   - Voice webhook: https://your-ngrok-url.ngrok.io/voice"
echo "   - Status callback: https://your-ngrok-url.ngrok.io/status"
echo ""
echo "Happy coding! 🎉"