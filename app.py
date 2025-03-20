import os
import threading
import time
import json
import uuid
import random
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

# Grammar categories to track
GRAMMAR_CATEGORIES = [
    "verb",         # ক্রিয়া
    "noun",         # বিশেষ্য
    "adjective",    # বিশেষণ
    "tense",        # কাল
    "punctuation",  # যতিচিহ্ন
    "syntax",       # বাক্য গঠন
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
            "weaknesses": {category: 0 for category in GRAMMAR_CATEGORIES},  # Track user's weaknesses
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

    # জেমিনি কে প্রম্পট তৈরি
    prompt = f"""**Role:** Act as a professional English teacher with 15 years experience.
**Task:** Check if the user's English translation matches the Bengali sentence.
**Instructions:**
1. Analyze spelling, grammar, and meaning accuracy
2. If incorrect, list errors in Bengali with detailed explanations
3. Always provide the correct translation

**Bengali:** {ban}
**User's Translation:** {eng}

**Output Format (STRICT JSON ONLY):**
Correct: {{
  "status": "correct",
  "message": "আপনার অনুবাদ সঠিক!",
  "correct_translation": "[সঠিক অনুবাদ]"
}}

Incorrect: {{
  "status": "incorrect",
  "message": "আপনার অনুবাদ সঠিক হয়নি।",
  "errors": {{
    "verb": "[ক্রিয়া ভুল]",
    "noun": "[বিশেষ্য ভুল]",
    "adjective": "[বিশেষণ ভুল]",
    "tense": "[কাল ভুল]",
    "punctuation": "[যতিচিহ্ন ভুল]",
    "syntax": "[বাক্য গঠন ভুল]"
  }},
  "why": {{
    "incorrect_reason": "[ভুলের কারণ বাংলায়]",
    "correction_explanation": "[সংশোধন সহ ব্যাখ্যা]"
  }},
  "correct_translation": "[সঠিক অনুবাদ]"
}}"""

    try:
        # জেমিনি থেকে রেসপন্স নিন
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # JSON ক্লিনআপ
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # JSON পার্স করুন
        json_response = json.loads(response_text)

        # Update user progress and history
        if user_id in user_sessions:
            user_session = user_sessions[user_id]
            history_entry = {
                'bengali': ban,
                'user_translation': eng,
                'correct': json_response.get('status') == 'correct',
                'errors': json_response.get('errors', {}),
                'correct_translation': json_response.get('correct_translation', ''),
                'timestamp': datetime.now().isoformat()
            }
            
            user_session['history'].append(history_entry)
            user_session['all_questions'].append(history_entry)  # Track all questions and answers
            
            # Update progress
            if history_entry['correct']:
                user_session['progress'] = min(user_session['progress'] + 2, 100)
            else:
                user_session['progress'] = max(user_session['progress'] - 1, 0)

            # Track weaknesses
            if not history_entry['correct']:
                for error_type, error_detail in history_entry['errors'].items():
                    if error_detail:  # If there is an error in this category
                        user_session['weaknesses'][error_type] += 1

            user_session['last_active'] = datetime.now()

        return jsonify(json_response)
        
    except json.JSONDecodeError:
        return jsonify({"error": "AI response format error", "raw": response_text}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get', methods=['GET'])
def generate_sentence():
    """যেকোনো লেভেলে বাংলা বাক্য জেনারেট করে এবং ইউজারের ইংরেজি শেখার অগ্রগতি অনুযায়ী বাক্য তৈরি করে"""
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
            "weaknesses": {category: 0 for category in GRAMMAR_CATEGORIES},  # ইউজারের দুর্বলতা ট্র্যাক করা
            "used_sentences": set(),  # ব্যবহৃত বাক্য ট্র্যাক করা
            "all_questions": [],  # সকল প্রশ্ন এবং উত্তর ট্র্যাক করা
        }

    user_session = user_sessions[user_id]
    
    # নতুন বাক্য তৈরির জন্য প্রম্পট তৈরি
    history_context = "Previous mistakes:\n"
    for entry in user_session['history'][-3:]:
        if not entry['correct']:
            history_context += f"- {entry['errors']}\n"

    # Focus on user's weaknesses
    weaknesses_context = "User's weaknesses:\n"
    for category, count in user_session['weaknesses'].items():
        if count > 0:
            weaknesses_context += f"- {category}: {count} errors\n"

    # Select a random sentence type and topic
    sentence_type = random.choice(SENTENCE_TYPES)
    topic = random.choice(TOPICS)

    prompt = f"""**User Profile:**
- Level: {level}
- Progress: {user_session['progress']}
- Recent errors: {history_context if history_context else 'None'}
- Weaknesses: {weaknesses_context if weaknesses_context else 'None'}
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
        # জেমিনি থেকে রেসপন্স নিন
        response = model.generate_content(prompt)
        sentence = response.text.strip(' "\n।') + '।'  # ফরম্যাট ঠিক করা

        # Ensure the sentence is unique
        while sentence in user_session['used_sentences']:
            response = model.generate_content(prompt)
            sentence = response.text.strip(' "\n।') + '।'

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
        "weaknesses": user_session['weaknesses'],
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
