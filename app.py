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
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

translation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 30,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

translation_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=translation_config,
)

# Store user sessions and their last active time
user_sessions = {}
tracking_codes = {}
tracking_lock = threading.Lock()

# Expanded list of sentence types for variety
SENTENCE_TYPES = [
    "প্রশ্নবোধক",      # Interrogative
    "বিস্ময়সূচক",     # Exclamatory
    "মজাদার",         # Funny
    "সরল",           # Simple/Normal
    "জটিল",          # Complex
    "অনুরোধমূলক",     # Request
    "উপদেশমূলক",      # Advice
    "অনুজ্ঞাসূচক",     # Imperative
    "বর্ণনামূলক",      # Descriptive
    "তুলনামূলক",      # Comparative
    "অনুমানমূলক",      # Hypothetical
    "ব্যঙ্গাত্মক",      # Sarcastic
    "ধাঁধাঁমূলক",       # Riddle-like
    "কবিতাসদৃশ",      # Poem-like
    "গানসদৃশ",        # Song-like
]

# Expanded list of topics with weights to prioritize daily life
TOPICS = [
    {"name": "দৈনন্দিন জীবন", "weight": 5},      # Daily life (higher weight)
    {"name": "শিক্ষা ও জ্ঞান", "weight": 4},     # Education and knowledge
    {"name": "খেলাধুলা ও বিনোদন", "weight": 3},  # Sports and entertainment
    {"name": "প্রকৃতি ও পরিবেশ", "weight": 3},   # Nature and environment
    {"name": "বিজ্ঞান ও প্রযুক্তি", "weight": 4},  # Science and technology
    {"name": "সাহিত্য ও সংস্কৃতি", "weight": 3},  # Literature and culture
    {"name": "ইতিহাস ও ঐতিহ্য", "weight": 2},    # History and heritage
    {"name": "ভ্রমণ ও পর্যটন", "weight": 3},    # Travel and tourism
    {"name": "স্বাস্থ্য ও চিকিৎসা", "weight": 4},  # Health and medicine
    {"name": "অর্থনীতি ও ব্যবসা", "weight": 2},   # Economy and business
    {"name": "রান্না ও খাদ্য", "weight": 5},     # Cooking and food (higher weight)
    {"name": "পোশাক ও ফ্যাশন", "weight": 3},   # Clothing and fashion
    {"name": "শিল্প ও কারুশিল্প", "weight": 2},   # Art and crafts
    {"name": "আন্তর্জাতিক সম্পর্ক", "weight": 2}, # International relations
    {"name": "সামাজিক সমস্যা", "weight": 3},    # Social issues
    {"name": "মনস্তত্ত্ব", "weight": 2},        # Psychology
    {"name": "দর্শন", "weight": 1},           # Philosophy
    {"name": "মহাকাশ ও জ্যোতির্বিদ্যা", "weight": 1}, # Space and astronomy
    {"name": "ভাষা ও ভাষাতত্ত্ব", "weight": 2},  # Language and linguistics
    {"name": "আইন ও বিচার", "weight": 1},      # Law and justice
]

def weighted_random_choice(choices):
    """Chooses a random element from a list of dictionaries with 'name' and 'weight' keys."""
    total_weight = sum(choice['weight'] for choice in choices)
    if total_weight == 0:
        return random.choice(choices)['name']
    random_num = random.uniform(0, total_weight)
    cumulative_weight = 0
    for choice in choices:
        cumulative_weight += choice['weight']
        if random_num < cumulative_weight:
            return choice['name']
    return choices[-1]['name']  # Should not happen, but for safety

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
            "sentence_type_usage": {},  # Track usage of each sentence type
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
        code_info = tracking_codes.get(tracking_code, None)

    if not code_info:
        return jsonify({"error": "Invalid or expired tracking code"}), 400

    ban = code_info['bengali']
    user_id = code_info['user_id']
    level = code_info['level']
    history = user_sessions.get(user_id, {}).get('all_questions', [])  # ব্যবহারকারীর অনুবাদ ইতিহাস

    # Enhanced prompt for more accurate translation checks with user history
    prompt = f"""**Role:** Act as a professional English teacher with 15 years of experience, specializing in Bengali to English translation. You are evaluating a user's translation as part of their language learning journey.

**User's Previous Translations:**
{[f"- Bengali: {item['bengali']}, User's Translation: {item['user_translation']}, Correct: {item['correct']}" for item in history[-5:]]}
(Showing last 5 translations for context)

**Task:** Carefully evaluate the user's English translation of the provided Bengali sentence.
**Instructions:**
1. **Comprehensive Analysis:** Check for accuracy in spelling, grammar, punctuation, word choice, and overall meaning. Consider nuances and idiomatic expressions. Pay attention to common mistakes the user might be making based on their previous translations.
2. **Constructive Feedback (in Bengali):** If the translation is incorrect, provide a detailed explanation in Bengali, highlighting the specific errors and why they are incorrect. Focus on areas where the user has struggled before.
3. **Provide Correct Translation:** Always include the accurate and natural-sounding English translation of the Bengali sentence.
4. **Strict JSON Output:** Ensure the output is a valid JSON object with the following structure:

{{
  "status": "correct" or "incorrect",
  "why": "[ভুলের কারণ বাংলায়]",
  "correct_translation": "[সঠিক অনুবাদ]"
}}

**Bengali Sentence:** {ban}
**User's Translation:** {eng}"""

    try:
        # Enhanced response handling with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = translation_model.generate_content(prompt)  # অনুবাদ মডেল ব্যবহার
                response_text = response.text.strip()

                # Attempt to extract JSON, handling potential leading/trailing text
                try:
                    start_index = response_text.find('{')
                    end_index = response_text.rfind('}')
                    if start_index != -1 and end_index != -1 and start_index < end_index:
                        json_string = response_text[start_index: end_index + 1]
                        json_response = json.loads(json_string)
                    else:
                        raise json.JSONDecodeError("No valid JSON found", response_text, 0)
                except json.JSONDecodeError as e:
                    if attempt == max_retries - 1:
                        return jsonify({
                            "error": "AI response format error",
                            "details": str(e),
                            "raw_response": response_text
                        }), 500
                    time.sleep(0.5)
                    continue

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
    history = user_sessions.get(user_id, {}).get('all_questions', [])  # ব্যবহারকারীর অনুবাদ ইতিহাস

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
            "sentence_type_usage": {},  # ট্র্যাক করে কোন ধরনের বাক্য কতবার ব্যবহার করা হয়েছে
        }

    user_session = user_sessions[user_id]

    # Determine the next sentence type, prioritizing less used types
    available_types = [
        st for st in SENTENCE_TYPES
        if user_session.get('sentence_type_usage', {}).get(st, 0) < len(user_session.get('all_questions', [])) // len(SENTENCE_TYPES) + 1
    ]
    if not available_types:
        available_types = SENTENCE_TYPES  # Fallback if all types have been used relatively equally

    sentence_type = random.choice(available_types)

    # Update sentence type usage count
    user_session.setdefault('sentence_type_usage', {}).setdefault(sentence_type, 0)
    user_session['sentence_type_usage'][sentence_type] += 1

    # Select a topic with weighted probability
    topic = weighted_random_choice(TOPICS)

    # Adjust prompt based on level
    if level <= 10:
        prompt = f"""**Role:** You are a helpful and experienced Bengali language tutor assisting a user in learning English. Your goal is to provide Bengali sentences that are appropriate for the user's current learning condition to help them improve their translation skills.

**User Profile:**
- Level: {level}
- Progress: {user_session['progress']}
- Previously used sentence types: {user_session.get('sentence_type_usage', {})}
- Current desired sentence type: {sentence_type}
- Topic: {topic}
- Recent Translation Performance: {[f"- Bengali: {item['bengali']}, Correct: {item['correct']}" for item in history[-3:]]} (Last 3 attempts)

**Task:** Generate a single Bengali sentence that is most suitable for the user based on their profile and recent performance. Consider the following:

1.  **Appropriateness:** The sentence should be very simple and easy to understand for a beginner.
2.  **Relevance:** The sentence should be relevant to the chosen topic ({topic}) and should aim to teach practical vocabulary and grammar.
3.  **Everyday Use Focus:** Prioritize sentences that reflect language commonly used in daily conversations in Bangladesh.
4.  **Sentence Type Variety:** Ensure the generated sentence aligns with the specified sentence type ({sentence_type}), contributing to a balanced learning experience across different types of sentences.
5.  **Learning Curve:** Gradually introduce more complex vocabulary, grammar, and idioms as the user's level and progress increase. Consider the user's recent success or failure in translations when deciding on complexity.
6.  **Uniqueness:** The sentence must be unique and should not have been presented to the user before (check against `user_session['used_sentences']`).
7.  **Natural and Idiomatic:** Ensure the Bengali sentence sounds natural and uses common Bengali idioms where appropriate for the level.

**Sentence Structure Guidelines:**
- **Vocabulary:** basic
- **Grammar:** simple
- **Length:** 5 to 10 words
- **Idioms:** no idioms
- **Style:** everyday use
- **Spelling:** simple

**Output Format:** Only the raw Bengali sentence without leading/trailing punctuation or quotes."""
    else:
        prompt = f"""**Role:** You are a helpful and experienced Bengali language tutor assisting a user in learning English. Your goal is to provide Bengali sentences that are appropriate for the user's current learning condition to help them improve their translation skills.

**User Profile:**
- Level: {level}
- Progress: {user_session['progress']}
- Previously used sentence types: {user_session.get('sentence_type_usage', {})}
- Current desired sentence type: {sentence_type}
- Topic: {topic}
- Recent Translation Performance: {[f"- Bengali: {item['bengali']}, Correct: {item['correct']}" for item in history[-3:]]} (Last 3 attempts)

**Task:** Generate a single Bengali sentence that is most suitable for the user based on their profile and recent performance. Consider the following:

1.  **Appropriateness:** The sentence should be at an appropriate difficulty level for the user ({'beginner' if level < 30 else 'intermediate' if level < 70 else 'advanced'}). It should offer a slight challenge, focusing on areas where they might be making mistakes.
2.  **Relevance:** The sentence should be relevant to the chosen topic ({topic}) and should aim to teach practical vocabulary and grammar.
3.  **Everyday Use Focus:** Prioritize sentences that reflect language commonly used in daily conversations in Bangladesh. However, also include sentences related to other important and useful topics (like international affairs, technology, etc.) with reasonable frequency.
4.  **Sentence Type Variety:** Ensure the generated sentence aligns with the specified sentence type ({sentence_type}), contributing to a balanced learning experience across different types of sentences.
5.  **Learning Curve:** Gradually introduce more complex vocabulary, grammar, and idioms as the user's level and progress increase. Consider the user's recent success or failure in translations when deciding on complexity.
6.  **Uniqueness:** The sentence must be unique and should not have been presented to the user before (check against `user_session['used_sentences']`).
7.  **Natural and Idiomatic:** Ensure the Bengali sentence sounds natural and uses common Bengali idioms where appropriate for the level.

**Sentence Structure Guidelines:**
- **Vocabulary:** {'basic' if level < 20 else 'simple' if level < 50 else 'intermediate'}
- **Grammar:** {'simple' if level < 30 else 'intermediate' if level < 70 else 'advanced'}
- **Length:** {level // 2 + 5} to {level // 2 + 15} words (adjust based on complexity).
- **Idioms:** Include {'no idioms' if level < 40 else '1 idiom' if level < 70 else '1-2 idioms'} (use naturally).
- **Style:** {'everyday use' if level < 50 else 'neutral' if level < 80 else 'slightly formal'} (match the context).
- **Spelling:** Use increasingly challenging spellings as the level increases.

**Output Format:** Only the raw Bengali sentence without leading/trailing punctuation or quotes."""

    try:
        # Generate sentence with retry mechanism
        max_retries = 5
        sentence = None
        for _ in range(max_retries):
            response = model.generate_content(prompt)
            if response.text:
                sentence = response.text.strip(' "\n।') + '।'
                if sentence not in user_session['used_sentences']:
                    break
            time.sleep(0.3)  # Add a small delay between retries

        if not sentence:
            return jsonify({"error": "Failed to generate a unique sentence"}), 500

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
        "sentence_type_usage": user_session.get('sentence_type_usage', {}),
    })

@app.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint to check if server is alive."""
    return jsonify({"status": "alive"})

def keep_alive():
    """Periodically pings the server to keep it alive."""
    url = "https://translate-vrv3.onrender.com/ping"  # Replace with your actual ping endpoint URL if different
    while True:
        time.sleep(300)  # প্রতি 5 মিনিট পর পিং করবে
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("✅ Keep-Alive Ping Successful")
            else:
                print(f"⚠️ Kee
