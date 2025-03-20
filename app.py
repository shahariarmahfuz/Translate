from flask import Flask, request, jsonify
import json
import re
import google.generativeai as genai  # সঠিক ইম্পোর্ট

app = Flask(__name__)

# সরাসরি API কী সেট করুন (প্রোডাকশনে এভয়েড করুন)
genai.configure(api_key="AIzaSyCRhglDw40RjOUEDjnVBWICC9zqO3oTcEY")  # 🔑 আপনার API কী দিয়ে রিপ্লেস করুন

def get_translation_feedback(ban_text, eng_text):
    model = "gemini-2.0-flash"  # মডেলের নাম আপডেট করুন

    contents = [
        # Correct translation example
        {
            "role": "user",
            "parts": [
                "বাংলা বাক্য: তিনি কোথায় যান?\nইংরেজি অনুবাদ: Where does he go?\n"
                "এই অনুবাদটি পরীক্ষা করে JSON ফরম্যাটে উত্তর দিন। কোনো অতিরিক্ত ব্যাখ্যা বা চিহ্ন ব্যবহার করবেন না।"
            ],
        },
        {
            "role": "model",
            "parts": [
                '{"status": "correct", "message": "আপনার অনুবাদ সঠিক!", "correct_translation": "Where does he go?"}'
            ],
        },
        
        # Incorrect translation example
        {
            "role": "user",
            "parts": [
                "বাংলা বাক্য: তিনি কোথায় যান?\nইংরেজি অনুবাদ: Whera duio he go?\n"
                "এই অনুবাদটি পরীক্ষা করে JSON ফরম্যাটে উত্তর দিন। কোনো অতিরিক্ত ব্যাখ্যা বা চিহ্ন ব্যবহার করবেন না।"
            ],
        },
        {
            "role": "model",
            "parts": [
                '{"status": "incorrect", "message": "আপনার অনুবাদ সঠিক হয়নি।", '
                '"errors": {"spelling": "Wrong spelling of \'Where\' and \'do\'.", "grammar": "Incorrect subject-verb agreement."}, '
                '"why": {"incorrect_reason": "\'Whera\' এবং \'duio\' শব্দ দুটি ভুল বানানে লেখা হয়েছে, এবং \'do\' ব্যবহৃত হয়েছে যেখানে \'does\' হওয়া উচিত।", '
                '"correction_explanation": "\'Where\' এবং \'do\' সঠিক বানানে লিখতে হবে এবং \'he\' তৃতীয় পুরুষ একবচন বিধায় \'does\' ব্যবহার করতে হবে। তাই সঠিক অনুবাদ হবে \'Where does he go?\'"}, '
                '"correct_translation": "Where does he go?"}'
            ],
        },
        
        # Current request
        {
            "role": "user",
            "parts": [
                f"বাংলা বাক্য: {ban_text}\nইংরেজি অনুবাদ: {eng_text}\n"
                "এই অনুবাদটি পরীক্ষা করে JSON ফরম্যাটে উত্তর দিন। কোনো অতিরিক্ত ব্যাখ্যা বা চিহ্ন ব্যবহার করবেন না।"
            ],
        }
    ]

    response = genai.generate_content(
        model=model,
        contents=contents,
        temperature=0.3,
        top_p=0.95,
        max_output_tokens=8192
    )

    # Clean JSON response
    clean_response = re.sub(r'^```json|```$', '', response.text, flags=re.IGNORECASE)
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
