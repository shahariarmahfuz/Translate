import os
import threading
import time
import json
import uuid
import random
import re
from datetime import datetime, timedelta
import requests
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Configure API key securely (should be set as an environment variable)
API_KEY = os.getenv("GENAI_API_KEY", "AIzaSyDaUj3swtOdBHSu-Jm_hP6nQuDiALHgsTY")
genai.configure(api_key=API_KEY)

# Set up the model with proper configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# Store user sessions and their last active time
user_sessions = {}
tracking_codes = {}
tracking_lock = threading.Lock()

SESSION_TIMEOUT = timedelta(hours=6)  # Set the session timeout to 6 hours
TRACKING_TIMEOUT = timedelta(hours=24)  # Set the tracking code timeout to 24 hours

# List of sentence types for variety
SENTENCE_TYPES = [
    "প্রশ্নবোধক",  # Interrogative
    "মজাদার",     # Fun
    "নরমাল",      # Normal
    "অনুরোধ",     # Request
    "আশ্চর্যবোধক", # Exclamatory
]

# List of topics for variety
TOPICS = [
    "দৈনন্দিন জীবন",  # Daily life
    "শিক্ষা",         # Education
    "খেলা",           # Sports
    "প্রকৃতি",        # Nature
    "প্রযুক্তি",      # Technology
    "সাহিত্য",        # Literature
]

@app.route("/ai", methods=["GET"])
def ai_response():
    """Handles AI response generation based on user input and session history."""
    question = request.args.get("q")
    user_id = request.args.get("id")

    if not question:
        return jsonify({"error": "Missing 'q' parameter"}), 400
    if not user_id:
        return jsonify({"error": "Missing 'id' parameter"}), 400

    # Initialize session history if user is new
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "history": [],
            "last_active": datetime.now(),
            "progress": 0,
            "used_sentences": set(),  # Track used sentences to avoid repetition
            "all_questions": [],  # Track all questions and answers
        }

    # Update last active time
    user_sessions[user_id]["last_active"] = datetime.now()

    # Append user message to history
    user_sessions[user_id]["history"].append({"role": "user", "parts": [question]})

    try:
        # Create chat session with user's history
        chat_session = model.start_chat(history=user_sessions[user_id]["history"])

        # Get AI response
        response = chat_session.send_message(question)

        if response.text:
            # Append AI response to history
            user_sessions[user_id]["history"].append({"role": "model", "parts": [response.text]})
            return jsonify({"response": response.text})
        else:
            return jsonify({"error": "AI did not return any response"}), 500

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/translate', methods=['GET'])
def translate_check():
    """যাচাই করে বাংলা থেকে ইংরেজি অনুবাদ সঠিক কিনা"""
    tracking_code = request.args.get('code')
    eng = request.args.get('en')
    
    # প্যারামিটার চেক
    if not tracking_code:
        return jsonify({"error": "Missing 'code' parameter"}), 400
    if not eng:
        return jsonify({"error": "Missing 'en' parameter"}), 400

    # Retrieve tracking code info
    with tracking_lock:
        code_info = tracking_codes.pop(tracking_code, None)
    
    if not code_info:
        return jsonify({"error": "Invalid or expired tracking code"}), 400

    ban = code_info['bengali']
    user_id = code_info['user_id']
    level = code_info['level']

    # Simplified prompt focusing only on why, correct_translation, and status
    prompt = f"""**Role:** Act as a professional English teacher with 15 years experience.
**Task:** Check if the user's English translation matches the Bengali sentence.
**Instructions:**
1. Analyze spelling, grammar, and meaning accuracy
2. If incorrect, explain why in Bengali
3. Always provide the correct translation

**Bengali:** {ban}
**User's Translation:** {eng}

**Output Format (STRICT JSON ONLY):**
{{
  "status": "correct/incorrect",
  "why": "[ভুলের কারণ বাংলায়]",
  "correct_translation": "[সঠিক অনুবাদ]"
}}"""

    try:
        # Enhanced response handling with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Extract JSON from response
                response_text = re.sub(r'^.*?{', '{', response_text, 1, re.DOTALL)
                response_text = re.sub(r'}.*?$', '}', response_text, 1, re.DOTALL)
                json_response = json.loads(response_text)

                # Validate response structure
                if 'status' not in json_response or 'why' not in json_response or 'correct_translation' not in json_response:
                    raise ValueError("Invalid response format")
                break
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries - 1:
                    return jsonify({
                        "error": "AI response format error",
                        "details": str(e),
                        "raw_response": response_text
                    }), 500
                time.sleep(0.5)

        # Update user progress
        if user_id in user_sessions:
            user_session = user_sessions[user_id]
            history_entry = {
                'bengali': ban,
                'user_translation': eng,
                'correct': json_response.get('status') == 'correct',
                'why': json_response.get('why', ''),
                'correct_translation': json_response.get('correct_translation', ''),
                'timestamp': datetime.now().isoformat()
            }
            
            user_session['all_questions'].append(history_entry)  # Track all questions and answers
            user_session['last_active'] = datetime.now()

        return jsonify(json_response)
        
    except Exception as e:
        return jsonify({
            "error": "Translation check failed",
            "details": str(e)
        }), 500

@app.route('/get', methods=['GET'])
def generate_sentence():
    """যেকোনো লেভেলে বাংলা বাক্য জেনারেট করে"""
    level = request.args.get('level', type=int)
    user_id = request.args.get('id')

    # ভ্যালিডেশন চেক
    if not level:
        return jsonify({"error": "Missing 'level' parameter"}), 400
    if not user_id:
        return jsonify({"error": "Missing 'id' parameter"}), 400
    if not 1 <= level <= 100:
        return jsonify({"error": "Level must be between 1 and 100"}), 400

    # ইউজার সেশন চেক করুন
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "history": [],
            "last_active": datetime.now(),
            "progress": 0,  # ইউজারের ইংরেজি শেখার অগ্রগতি
            "used_sentences": set(),  # ব্যবহৃত বাক্য ট্র্যাক করা
            "all_questions": [],  # সকল প্রশ্ন এবং উত্তর ট্র্যাক করা
        }

    user_session = user_sessions[user_id]
    
    # Select a random sentence type and topic
    sentence_type = random.choice(SENTENCE_TYPES)
    topic = random.choice(TOPICS)

    prompt = f"""**User Profile:**
- Level: {level}
- Progress: {user_session['progress']}
- Sentence type: {sentence_type}
- Topic: {topic}

Generate a Bengali sentence focusing on:
1. Use {'basic' if level <20 else 'simple'} vocabulary
2. Include {'simple' if level <30 else 'intermediate'} grammar
3. Length: {level//2 +5} to {level//2 +15} words
4. Add {'no idioms' if level <40 else '1-2 idioms'} 
5. Make it {'everyday use' if level <50 else 'neutral'} 
6. Use challenging spellings as level increases
7. Consider the user's progress in English while creating the sentence.

**Output Format:** Only the raw Bengali sentence without punctuation/quotes"""
    
    try:
        # Generate sentence with retry mechanism
        max_retries = 5
        sentence = None
        for _ in range(max_retries):
            response = model.generate_content(prompt)
            sentence = response.text.strip(' "\n।') + '।'
            if sentence not in user_session['used_sentences']:
                break
        
        if not sentence:
            return jsonify({"error": "Failed to generate unique sentence"}), 500

        # Mark the sentence as used
        user_session['used_sentences'].add(sentence)

        # Generate tracking code
        tracking_code = uuid.uuid4().hex
        with tracking_lock:
            tracking_codes[tracking_code] = {
                'bengali': sentence,
                'user_id': user_id,
                'level': level,
                'timestamp': datetime.now()
            }

        return jsonify({
            "sentence": sentence,
            "tracking_code": tracking_code,
            "progress": user_session['progress'],
            "sentence_type": sentence_type,
            "topic": topic,
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/progress', methods=['GET'])
def get_progress():
    """ইউজারের প্রোগ্রেস এবং দুর্বলতা রিপোর্ট দেখায়"""
    user_id = request.args.get('id')

    if not user_id:
        return jsonify({"error": "Missing 'id' parameter"}), 400

    if user_id not in user_sessions:
        return jsonify({"error": "User not found"}), 404

    user_session = user_sessions[user_id]
    return jsonify({
        "progress": user_session['progress'],
        "all_questions": user_session['all_questions'],
    })

@app.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint to check if server is alive."""
    return jsonify({"status": "alive"})

def clean_resources():
    """Periodically checks and removes inactive user sessions and expired tracking codes."""
    while True:
        now = datetime.now()
        
        # Clean user sessions
        for user_id in list(user_sessions.keys()):
            if now - user_sessions[user_id]['last_active'] > SESSION_TIMEOUT:
                print(f"🧹 Removing inactive session for user {user_id}")
                del user_sessions[user_id]
                
        # Clean tracking codes
        with tracking_lock:
            for code in list(tracking_codes.keys()):
                if now - tracking_codes[code]['timestamp'] > TRACKING_TIMEOUT:
                    print(f"🧹 Removing expired tracking code {code}")
                    del tracking_codes[code]
        
        time.sleep(300)  # Check every 5 minutes

def keep_alive():
    """Periodically pings the server to keep it alive."""
    url = "https://new-ai-buxr.onrender.com/ping"
    while True:
        time.sleep(300)  # প্রতি 5 মিনিট পর পিং করবে
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("✅ Keep-Alive Ping Successful")
            else:
                print(f"⚠️ Keep-Alive Ping Failed: Status Code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Keep-Alive Error: {e}")

# Run clean-up and keep-alive in separate threads
clean_up_thread = threading.Thread(target=clean_resources, daemon=True)
clean_up_thread.start()

keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
keep_alive_thread.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
