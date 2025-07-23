#!/usr/bin/env python3
"""
Twilio Voice Application
Handles incoming voice calls with TwiML responses
"""

from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')


@app.route('/voice', methods=['GET', 'POST'])
def voice():
    """Handle incoming voice calls"""
    response = VoiceResponse()
    
    # Gather input from the caller
    gather = Gather(
        num_digits=1,
        action='/gather',
        method='POST',
        timeout=10
    )
    
    gather.say(
        "Welcome to our voice application. "
        "Press 1 for sales. "
        "Press 2 for support. "
        "Press 3 to leave a message.",
        voice='alice',
        language='en-US'
    )
    
    response.append(gather)
    
    # If no input, repeat the menu
    response.redirect('/voice')
    
    return Response(str(response), mimetype='text/xml')


@app.route('/gather', methods=['POST'])
def gather():
    """Process gathered digits from the caller"""
    response = VoiceResponse()
    
    # Get the digit pressed by the caller
    if 'Digits' in request.values:
        choice = request.values['Digits']
        
        if choice == '1':
            response.say(
                "Connecting you to our sales team. Please hold.",
                voice='alice'
            )
            # Forward to sales number
            response.dial('+1234567890')  # Replace with actual sales number
            
        elif choice == '2':
            response.say(
                "Connecting you to support. Please hold.",
                voice='alice'
            )
            # Forward to support number
            response.dial('+1234567891')  # Replace with actual support number
            
        elif choice == '3':
            response.say(
                "Please leave your message after the beep. Press the pound key when finished.",
                voice='alice'
            )
            # Record the caller's message
            response.record(
                maxLength=60,
                action='/recording',
                finishOnKey='#'
            )
            
        else:
            response.say(
                "Invalid selection. Returning to the main menu.",
                voice='alice'
            )
            response.redirect('/voice')
    else:
        # No input received
        response.redirect('/voice')
    
    return Response(str(response), mimetype='text/xml')


@app.route('/recording', methods=['POST'])
def recording():
    """Handle recorded messages"""
    response = VoiceResponse()
    
    # Get the recording URL
    recording_url = request.values.get('RecordingUrl', '')
    
    response.say(
        "Thank you for your message. We'll get back to you soon. Goodbye!",
        voice='alice'
    )
    response.hangup()
    
    # Log the recording URL (in production, save to database or send notification)
    if recording_url:
        print(f"New recording received: {recording_url}")
    
    return Response(str(response), mimetype='text/xml')


@app.route('/status', methods=['POST'])
def status_callback():
    """Handle call status callbacks"""
    call_sid = request.values.get('CallSid', '')
    call_status = request.values.get('CallStatus', '')
    
    print(f"Call {call_sid} status: {call_status}")
    
    return Response('', status=200)


@app.route('/')
def index():
    """Basic index page"""
    return '''
    <h1>Twilio Voice Application</h1>
    <p>This application handles incoming voice calls.</p>
    <ul>
        <li>POST /voice - Main voice webhook</li>
        <li>POST /gather - Process user input</li>
        <li>POST /recording - Handle recordings</li>
        <li>POST /status - Call status callbacks</li>
    </ul>
    '''


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)