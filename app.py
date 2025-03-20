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
    try:
        tracking_code = request.args.get('code')
        eng = request.args.get('en')
        
        if not tracking_code or not eng:
            return jsonify({"error": "Missing parameters"}), 400

        with tracking_lock:
            code_info = tracking_codes.pop(tracking_code, None)
        
        if not code_info:
            return jsonify({"error": "Invalid or expired tracking code"}), 400

        ban = code_info['bengali']
        user_id = code_info['user_id']
        level = code_code['level']

        # উন্নত প্রম্পট ডিজাইন
        prompt = f"""**Role:** Senior English Language Examiner
**Task:** Comprehensive Translation Validation
**Bengali Sentence:** {ban}
**User Translation:** {eng}

**Analysis Criteria:**
1. Spelling Check (বানান যাচাই)
2. Grammar Check (ব্যাকরণ যাচাই) - {GRAMMAR_CATEGORIES}
3. Semantic Accuracy (অর্থের যথার্থতা)
4. Context Preservation (প্রাসঙ্গিকতা)

**Error Types to Identify:**
- Spelling Mistake (বানান ভুল)
- Verb Form Error (ক্রিয়া রূপ)
- Tense Mismatch (কালের অমিল)
- Preposition Error (পদান্বয়ী অব্যয়)
- Word Order Issue (শব্দ বিন্যাস)
- Punctuation Error (যতিচিহ্ন)
- Articles (a/an/the) Usage
- Plural/Singular Form (একবচন/বহুবচন)

**Response Format (STRICT JSON):**
{{
  "status": "correct|incorrect",
  "message": "বিস্তারিত ফিডব্যাক",
  "errors": {{
    "spelling": ["ভুল বানান", "সঠিক বানান"],
    "grammar": {{
      "category1": "ব্যাখ্যা",
      "category2": "ব্যাখ্যা"
    }},
    "semantic": "অর্থগত পার্থক্য",
    "context": "প্রাসঙ্গিকতা হারানো"
  }},
  "correct_translation": "সঠিক অনুবাদ"
}}"""

        response = model.generate_content(prompt)
        
        # JSON ভ্যালিডেশন উন্নত করা হয়েছে
        try:
            json_response = json.loads(response.text.strip('```json\n'))
        except json.JSONDecodeError:
            # ফলব্যাক পার্সিং লজিক
            error_msg = response.text.lower()
            errors = {}
            
            if 'spelling' in error_msg:
                errors['spelling'] = ["বানান ভুল", ""]
            if 'verb' in error_msg:
                errors['grammar'] = {"verb": "ক্রিয়া রূপ ভুল"}
            if 'tense' in error_msg:
                errors['grammar'] = {"tense": "কালের অমিল"}
            
            json_response = {
                "status": "incorrect",
                "message": "ত্রুটি শনাক্ত করা হয়েছে",
                "errors": errors,
                "correct_translation": ""
            }

        # ইউজার সেশন আপডেট
        if user_id in user_sessions:
            user_session = user_sessions[user_id]
            
            # এরর ক্যাটাগরাইজেশন
            detailed_errors = {
                'spelling': json_response.get('errors', {}).get('spelling', []),
                'grammar': json_response.get('errors', {}).get('grammar', {}),
                'semantic': json_response.get('errors', {}).get('semantic', ''),
                'context': json_response.get('errors', {}).get('context', '')
            }

            history_entry = {
                'bengali': ban,
                'user_translation': eng,
                'correct': json_response.get('status') == 'correct',
                'errors': detailed_errors,
                'correct_translation': json_response.get('correct_translation', ''),
                'timestamp': datetime.now().isoformat()
            }

            # প্রোগ্রেস আপডেট লজিক
            if history_entry['correct']:
                user_session['progress'] += 2
            else:
                user_session['progress'] -= 1
                
                # স্পেসিফিক এরর ট্র্যাকিং
                if detailed_errors['spelling']:
                    user_session['weaknesses']['spelling'] += 2
                for gram_error in detailed_errors['grammar']:
                    user_session['weaknesses'][gram_error] += 1

            user_session['history'].append(history_entry)
            user_session['last_active'] = datetime.now()

        return jsonify(json_response)
        
    except Exception as e:
        logging.error(f"Translation Error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "পরীক্ষা প্রক্রিয়ায় সমস্যা",
            "technical": f"{type(e).__name__}: {str(e)}"
        }), 500
        
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
