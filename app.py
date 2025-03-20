from flask import Flask, request, jsonify
import google.generativeai as genai  # ✅ সঠিক ইম্পোর্ট মেথড
import os
import re
import json

app = Flask(__name__)

# Configure GenAI API
genai.configure(api_key="AIzaSyCRhglDw40RjOUEDjnVBWICC9zqO3oTcEY")  # 🔑 আপনার API কী দিয়ে রিপ্লেস করুন

def get_translation_feedback(ban_text, eng_text):
    model = "gemini-2.0-flash"

    contents = [
        # Correct translation example
        genai.Content(
            role="user",
            parts=[genai.Part.from_text(
                "বাংলা বাক্য: তিনি কোথায় যান?\nইংরেজি অনুবাদ: Where does he go?\n"
                "এই অনুবাদটি পরীক্ষা করে JSON ফরম্যাটে উত্তর দিন। কোনো অতিরিক্ত ব্যাখ্যা বা চিহ্ন ব্যবহার করবেন না।"
            )],
        ),
        genai.Content(
            role="model",
            parts=[genai.Part.from_text(
                '{"status": "correct", "message": "আপনার অনুবাদ সঠিক!", "correct_translation": "Where does he go?"}'
            )],
        ),
        
        # Incorrect translation example
        genai.Content(
            role="user",
            parts=[genai.Part.from_text(
                "বাংলা বাক্য: তিনি কোথায় যান?\nইংরেজি অনুবাদ: Whera duio he go?\n"
                "এই অনুবাদটি পরীক্ষা করে JSON ফরম্যাটে উত্তর দিন। কোনো অতিরিক্ত ব্যাখ্যা বা চিহ্ন ব্যবহার করবেন না।"
            )],
        ),
        genai.Content(
            role="model",
            parts=[genai.Part.from_text(
                '{"status": "incorrect", "message": "আপনার অনুবাদ সঠিক হয়নি।", '
                '"errors": {"spelling": "Wrong spelling of \'Where\' and \'do\'.", "grammar": "Incorrect subject-verb agreement."}, '
                '"why": {"incorrect_reason": "\'Whera\' এবং \'duio\' শব্দ দুটি ভুল বানানে লেখা হয়েছে, এবং \'do\' ব্যবহৃত হয়েছে যেখানে \'does\' হওয়া উচিত।", '
                '"correction_explanation": "\'Where\' এবং \'do\' সঠিক বানানে লিখতে হবে এবং \'he\' তৃতীয় পুরুষ একবচন বিধায় \'does\' ব্যবহার করতে হবে। তাই সঠিক অনুবাদ হবে \'Where does he go?\'"}, '
                '"correct_translation": "Where does he go?"}'
            )],
        ),
        
        # Current request
        genai.Content(
            role="user",
            parts=[genai.Part.from_text(
                f"বাংলা বাক্য: {ban_text}\nইংরেজি অনুবাদ: {eng_text}\n"
                "এই অনুবাদটি পরীক্ষা করে JSON ফরম্যাটে উত্তর দিন। কোনো অতিরিক্ত ব্যাখ্যা বা চিহ্ন ব্যবহার করবেন না।"
            )],
        )
    ]

    config = genai.GenerateContentConfig(
        temperature=0.3,
        top_p=0.95,
        max_output_tokens=8192,
        response_mime_type="text/plain"
    )

    # Assuming `generate_content_stream` is the correct method
    response = genai.generate_content_stream(
        model=model,
        contents=contents,
        config=config
    )

    full_response = ""
    for chunk in response:
        if chunk.text:
            full_response += chunk.text

    # Clean JSON response
    clean_response = re.sub(r'^```json|```$', '', full_response, flags=re.IGNORECASE)
    return clean_response.strip()

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
