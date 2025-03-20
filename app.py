# app.py
from flask import Flask, request, jsonify
import google.generativeai as genai  # ✅ সঠিক ইম্পোর্ট মেথড
import os
import re
import json
from google.generativeai import types  # Import types explicitly

app = Flask(__name__)

# Configure GenAI API
genai.configure(api_key="AIzaSyCRhglDw40RjOUEDjnVBWICC9zqO3oTcEY")  # 🔑 আপনার API কী দিয়ে রিপ্লেস করুন
model = genai.GenerativeModel('gemini-pro') # Initialize the generative model

def get_translation_feedback(ban_text, eng_text):
    contents = [
        # Correct translation example
        types.Content(
            role="user",
            parts=[types.Part.from_text(
                "বাংলা বাক্য: তিনি কোথায় যান?\nইংরেজি অনুবাদ: Where does he go?\n"
                "এই অনুবাদটি পরীক্ষা করে JSON ফরম্যাটে উত্তর দিন। কোনো অতিরিক্ত ব্যাখ্যা বা চিহ্ন ব্যবহার করবেন না।"
            )],
        ),
        types.Content(
            role="model",
            parts=[types.Part.from_text(
                '{"status": "correct", "message": "আপনার অনুবাদ সঠিক!", "correct_translation": "Where does he go?"}'
            )],
        ),

        # Incorrect translation example
        types.Content(
            role="user",
            parts=[types.Part.from_text(
                "বাংলা বাক্য: তিনি কোথায় যান?\nইংরেজি অনুবাদ: Whera duio he go?\n"
                "এই অনুবাদটি পরীক্ষা করে JSON ফরম্যাটে উত্তর দিন। কোনো অতিরিক্ত ব্যাখ্যা বা চিহ্ন ব্যবহার করবেন না।"
            )],
        ),
        types.Content(
            role="model",
            parts=[types.Part.from_text(
                '{"status": "incorrect", "message": "আপনার অনুবাদ সঠিক হয়নি।", '
                '"errors": {"spelling": "Wrong spelling of \'Where\' and \'do\'.", "grammar": "Incorrect subject-verb agreement."}, '
                '"why": {"incorrect_reason": "\'Whera\' এবং \'duio\' শব্দ দুটি ভুল বানানে লেখা হয়েছে, এবং \'do\' ব্যবহৃত হয়েছে যেখানে \'does\' হওয়া উচিত।", '
                '"correction_explanation": "\'Where\' এবং \'do\' সঠিক বানানে লিখতে হবে এবং \'he\' তৃতীয় পুরুষ একবচন বিধায় \'does\' ব্যবহার করতে হবে। তাই সঠিক অনুবাদ হবে \'Where does he go?\'"}, '
                '"correct_translation": "Where does he go?"}'
            )],
        ),

        # Current request
        types.Content(
            role="user",
            parts=[types.Part.from_text(
                f"বাংলা বাক্য: {ban_text}\nইংরেজি অনুবাদ: {eng_text}\n"
                "এই অনুবাদটি পরীক্ষা করে JSON ফরম্যাটে উত্তর দিন। কোনো অতিরিক্ত ব্যাখ্যা বা চিহ্ন ব্যবহার করবেন না।"
            )],
        )
    ]

    response = model.generate_content(contents)
    return response.text.strip()

@app.route('/translate', methods=['GET'])
def handle_translation():
    ban_text = request.args.get('ban')
    eng_text = request.args.get('eng')

    if not ban_text or not eng_text:
        return jsonify({"error": "Missing 'ban' or 'eng' parameters"}), 400

    try:
        gemini_response = get_translation_feedback(ban_text, eng_text)
        return jsonify(json.loads(gemini_response))
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid response format from AI model"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
