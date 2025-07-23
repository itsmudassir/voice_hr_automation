# Twilio Voice Application

A complete Twilio voice application with both inbound and outbound calling capabilities.

## Features

### Inbound Calls (app.py)
- Interactive Voice Response (IVR) menu
- Call routing to different departments
- Voice message recording
- Call status tracking

### Outbound Calls (outbound_app.py)
- API-driven outbound calling
- Custom message delivery
- Call response tracking
- Call management endpoints

## Setup

### 1. Prerequisites
- Python 3.8 or higher
- Twilio account with phone number
- ngrok (for local testing)

### 2. Installation

Run the setup script:
```bash
./setup.sh
```

Or manually:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Update `.env` file with your Twilio credentials:
```env
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890
```

### 4. Running the Applications

#### Inbound Call Handler:
```bash
python app.py
```

#### Outbound Call API:
```bash
python outbound_app.py
```

### 5. Local Testing with ngrok

1. Install ngrok from https://ngrok.com
2. Start ngrok:
   ```bash
   ngrok http 5000
   ```
3. Use the HTTPS URL for Twilio webhooks

## API Endpoints

### Inbound App (Port 5000)
- `POST /voice` - Main voice webhook
- `POST /gather` - Process user input
- `POST /recording` - Handle recordings
- `POST /status` - Call status callbacks

### Outbound App (Port 5001)
- `POST /make-call` - Initiate outbound call
- `GET /list-calls` - List all calls
- `GET /call-details/{call_sid}` - Get call details

## Making an Outbound Call

```bash
curl -X POST http://localhost:5001/make-call \
  -H "Content-Type: application/json" \
  -d '{
    "to": "+1234567890",
    "message": "Hello, this is a test call from our system."
  }'
```

## Twilio Console Configuration

1. Go to your Twilio phone number settings
2. Set the voice webhook to: `https://your-ngrok-url.ngrok.io/voice`
3. Set the status callback to: `https://your-ngrok-url.ngrok.io/status`

## Project Structure

```
voice/
├── app.py              # Inbound call handler
├── outbound_app.py     # Outbound call API
├── requirements.txt    # Python dependencies
├── requirements_mcp.txt # MCP dependencies (for GitHub integration)
├── setup.sh           # Setup script
├── .env               # Environment variables (create this)
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## Security Notes

- Never commit `.env` file to version control
- Use environment variables for all sensitive data
- Validate phone numbers before making calls
- Implement rate limiting in production
- Use HTTPS for all webhooks

## Troubleshooting

### Common Issues

1. **"Twilio could not find a valid TwiML response"**
   - Ensure your webhook URL is publicly accessible
   - Check that your app is returning valid TwiML

2. **Calls not connecting**
   - Verify phone numbers are in E.164 format (+1234567890)
   - Check Twilio account has sufficient balance
   - Ensure phone number has voice capabilities

3. **ngrok connection issues**
   - Restart ngrok and update webhook URLs
   - Check firewall settings

## License

This project is provided as-is for educational purposes.