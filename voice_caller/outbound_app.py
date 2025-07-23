#!/usr/bin/env python3
"""
Twilio Outbound Voice Call Application
Makes outbound calls and handles call flows
"""

from flask import Flask, request, Response, jsonify
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Store active calls (in production, use a database)
active_calls = {}


@app.route('/make-call', methods=['POST'])
def make_call():
    """Initiate an outbound call"""
    data = request.get_json()
    
    if not data or 'to' not in data:
        return jsonify({'error': 'Missing "to" phone number'}), 400
    
    to_number = data['to']
    message = data.get('message', 'Hello, this is a test call.')
    
    try:
        # Make the outbound call
        call = client.calls.create(
            url=request.url_root + 'outbound-voice',
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            status_callback=request.url_root + 'call-status',
            status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
            status_callback_method='POST'
        )
        
        # Store call information
        active_calls[call.sid] = {
            'to': to_number,
            'message': message,
            'created_at': datetime.now().isoformat(),
            'status': 'initiated'
        }
        
        return jsonify({
            'success': True,
            'call_sid': call.sid,
            'to': to_number,
            'status': 'initiated'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/outbound-voice', methods=['POST'])
def outbound_voice():
    """Handle the outbound call flow"""
    response = VoiceResponse()
    
    # Get call SID to retrieve custom message
    call_sid = request.values.get('CallSid', '')
    call_info = active_calls.get(call_sid, {})
    message = call_info.get('message', 'Hello, this is an automated call.')
    
    # Initial greeting
    response.say(
        message,
        voice='alice',
        language='en-US'
    )
    
    # Gather input
    gather = Gather(
        num_digits=1,
        action='/outbound-gather',
        method='POST',
        timeout=10
    )
    
    gather.say(
        "Press 1 to confirm receipt of this message. "
        "Press 2 to speak with a representative. "
        "Press 3 to be removed from our calling list.",
        voice='alice'
    )
    
    response.append(gather)
    
    # If no input, say goodbye
    response.say("We did not receive a response. Goodbye!", voice='alice')
    response.hangup()
    
    return Response(str(response), mimetype='text/xml')


@app.route('/outbound-gather', methods=['POST'])
def outbound_gather():
    """Process gathered digits from outbound calls"""
    response = VoiceResponse()
    
    if 'Digits' in request.values:
        choice = request.values['Digits']
        
        if choice == '1':
            response.say(
                "Thank you for confirming. Have a great day!",
                voice='alice'
            )
            response.hangup()
            
        elif choice == '2':
            response.say(
                "Connecting you to a representative. Please hold.",
                voice='alice'
            )
            # Connect to a representative
            response.dial('+1234567890')  # Replace with actual number
            
        elif choice == '3':
            response.say(
                "You have been removed from our calling list. Goodbye!",
                voice='alice'
            )
            # In production, add to do-not-call list
            response.hangup()
            
        else:
            response.say(
                "Invalid selection. Goodbye!",
                voice='alice'
            )
            response.hangup()
    else:
        response.say("We did not receive a response. Goodbye!", voice='alice')
        response.hangup()
    
    return Response(str(response), mimetype='text/xml')


@app.route('/call-status', methods=['POST'])
def call_status():
    """Handle call status updates"""
    call_sid = request.values.get('CallSid', '')
    call_status = request.values.get('CallStatus', '')
    
    # Update call information
    if call_sid in active_calls:
        active_calls[call_sid]['status'] = call_status
        active_calls[call_sid]['last_updated'] = datetime.now().isoformat()
    
    print(f"Call {call_sid} status updated: {call_status}")
    
    # Clean up completed calls (in production, move to database)
    if call_status in ['completed', 'failed', 'busy', 'no-answer']:
        if call_sid in active_calls:
            print(f"Call completed: {active_calls[call_sid]}")
            # In production, save to database instead of deleting
            # del active_calls[call_sid]
    
    return Response('', status=200)


@app.route('/list-calls', methods=['GET'])
def list_calls():
    """List all active and recent calls"""
    return jsonify({
        'calls': active_calls,
        'total': len(active_calls)
    })


@app.route('/call-details/<call_sid>', methods=['GET'])
def call_details(call_sid):
    """Get details for a specific call"""
    if call_sid in active_calls:
        return jsonify(active_calls[call_sid])
    else:
        return jsonify({'error': 'Call not found'}), 404


@app.route('/')
def index():
    """API documentation"""
    return '''
    <h1>Twilio Outbound Call API</h1>
    <h2>Endpoints:</h2>
    <ul>
        <li><strong>POST /make-call</strong> - Initiate an outbound call
            <pre>
{
    "to": "+1234567890",
    "message": "Your custom message here"
}
            </pre>
        </li>
        <li><strong>GET /list-calls</strong> - List all calls</li>
        <li><strong>GET /call-details/{call_sid}</strong> - Get specific call details</li>
        <li><strong>POST /outbound-voice</strong> - Webhook for call flow (Twilio use)</li>
        <li><strong>POST /outbound-gather</strong> - Webhook for user input (Twilio use)</li>
        <li><strong>POST /call-status</strong> - Webhook for status updates (Twilio use)</li>
    </ul>
    '''


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)