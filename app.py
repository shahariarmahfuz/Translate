import os
import threading
import time
import json
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

SESSION_TIMEOUT = timedelta(hours=6)  # Set the session timeout to 6 hours

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
            "last_active": datetime.now()
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
    ban = request.args.get('ban')
    eng = request.args.get('eng')
    
    # প্যারামিটার চেক
    if not ban:
        return jsonify({"error": "Missing 'ban' parameter"}), 400
    if not eng:
        return jsonify({"error": "Missing 'eng' parameter"}), 400

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
    "spelling": "[বানান ভুল]",
    "grammar": "[ব্যাকরণ ভুল]"
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
            "progress": 0  # ইউজারের ইংরেজি শেখার অগ্রগতি
        }

    # ইউজারের পূর্ববর্তী অগ্রগতি চেক করা
    progress = user_sessions[user_id]["progress"]
    
    # নতুন বাক্য তৈরির জন্য প্রম্পট তৈরি
    if level < 50:
        prompt = f"""**Role:** Act as a professional English teacher.  
**Task:** Generate a Bengali sentence for English translation practice based on the user's progress in learning English.
**User's English Progress:** {progress}/100 (0 = Beginner, 100 = Fluent)
**Difficulty Level:** {level}/100 (1=easiest, 100=hardest)
**Requirements:**
1. Use {'basic' if level <20 else 'simple'} vocabulary
2. Include {'simple' if level <30 else 'intermediate'} grammar
3. Length: {level//2 +5} to {level//2 +15} words
4. Add {'no idioms' if level <40 else '1-2 idioms'} 
5. Make it {'everyday use' if level <50 else 'neutral'} 
6. Use challenging spellings as level increases
7. Consider the user's progress in English while creating the sentence.

**Output Format:** Only the raw Bengali sentence without punctuation/quotes"""
    else:
        prompt = f"""**Role:** Act as a professional English teacher.  
**Task:** Generate a Bengali sentence for English translation practice based on the user's progress in learning English.
**User's English Progress:** {progress}/100 (0 = Beginner, 100 = Fluent)
**Difficulty Level:** {level}/100 (1=easiest, 100=hardest)
**Requirements:**
1. Use {'common' if level <70 else 'advanced'} vocabulary
2. Include {'intermediate' if level <70 else 'complex'} grammar
3. Length: {level//2 +5} to {level//2 +15} words
4. Add {'1-2 idioms' if level <80 else 'more complex expressions'} 
5. Make it {'neutral' if level <80 else 'technical/professional'} 
6. Use challenging spellings and expressions
7. Consider the user's progress in English while creating the sentence.

**Output Format:** Only the raw Bengali sentence without punctuation/quotes"""
    
    try:
        # জেমিনি থেকে রেসপন্স নিন
        response = model.generate_content(prompt)
        sentence = response.text.strip(' "\n।') + '।'  # ফরম্যাট ঠিক করা

        # ইউজার সেশনে বাক্য সংরক্ষণ করুন
        user_sessions[user_id]["history"].append(sentence)
        user_sessions[user_id]["last_active"] = datetime.now()

        # ইউজারের ইংরেজি শেখার অগ্রগতি আপডেট করুন (এটি কোনো প্রোগ্রেস ট্র্যাকিং মেকানিজম হতে পারে)
        user_sessions[user_id]["progress"] += 1  # উদাহরণস্বরূপ, প্রোগ্রেস আপডেট করা হচ্ছে। এটি আরও কাস্টমাইজ করা যেতে পারে।

        return jsonify({"sentence": sentence})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint to check if server is alive."""
    return jsonify({"status": "alive"})

def clean_inactive_sessions():
    """Periodically checks and removes inactive user sessions."""
    while True:
        current_time = datetime.now()
        for user_id, session_data in list(user_sessions.items()):
            if current_time - session_data["last_active"] > SESSION_TIMEOUT:
                print(f"🧹 Removing inactive session for user {user_id}")
                del user_sessions[user_id]
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
clean_up_thread = threading.Thread(target=clean_inactive_sessions, daemon=True)
clean_up_thread.start()

keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
keep_alive_thread.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
