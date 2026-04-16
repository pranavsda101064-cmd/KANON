import codecs
import re

with codecs.open('main.py', 'r', 'utf-8') as f:
    text = f.read()

# 1. Imports
if 'import google.generativeai as genai' not in text:
    text = text.replace('import torch', 'import torch\nimport google.generativeai as genai')

# 2. Setup Gemini
init_gemini = """
# Initialize Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
chat_sessions = {}
SYSTEM_INSTRUCTION = \"\"\"You are Dr. KANON AI, a highly advanced medical AI designed for rural South India.
You must speak in the language the user speaks to you (or as requested).
Ask follow up questions back-and-forth like a real doctor consultation to understand their symptoms.
Keep responses empathetic but concise.
Once you have gathered enough symptom context to confidently offer a potential diagnosis and remedies, your response MUST begin exactly with:
[DIAGNOSIS_FINAL]
Followed by the structured assessment, the suspected conditions, warning signs, and home remedies.
Do not use [DIAGNOSIS_FINAL] until the consultation is complete.
\"\"\"

def get_chat_session(session_id: str):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = gemini_model.start_chat(history=[])
        chat_sessions[session_id].send_message(SYSTEM_INSTRUCTION)
    return chat_sessions[session_id]
"""

if 'import google.generativeai as genai' in init_gemini or 'gemini_model' not in text:
    text = text.replace('medmo_model = None\n\n', 'medmo_model = None\n\n' + init_gemini + '\n')

# 3. Replace analyze_symptoms with chat endpoints
if '/analyze/symptoms' in text:
    pattern_endpoints = re.compile(r'@app\.post\("/analyze/symptoms"\).*?return JSONResponse.*?status_code=500\)', re.DOTALL)

    gemini_endpoints = """@app.post("/chat/message")
async def chat_message(
    session_id: str = Form(...),
    message: str = Form(...),
    language: str = Form("en")
):
    try:
        # Check API key config
        if not genai.config.api_key or genai.config.api_key == "YOUR_GEMINI_API_KEY":
            return JSONResponse({"success": False, "error": "INVALID_API_KEY", "message": "Gemini API key is missing. Please add it to main.py."})
            
        chat = get_chat_session(session_id)
        
        prompt = f"[{language}] {message}"
        response = chat.send_message(prompt)
        reply_text = response.text.strip()
        
        is_final = False
        diagnosis = None
        turn = len(chat.history) // 2
        
        if "[DIAGNOSIS_FINAL]" in reply_text:
            is_final = True
            parts = reply_text.split("[DIAGNOSIS_FINAL]")
            reply_text = parts[0].strip() if parts[0].strip() else "Here is my final assessment based on our consultation."
            diagnosis = parts[1].strip()
            
        return JSONResponse({
            "success": True,
            "message": reply_text,
            "is_final": is_final,
            "diagnosis": diagnosis,
            "turn": turn
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/chat/reset")
async def chat_reset(session_id: str = Form(...)):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return JSONResponse({"success": True})"""

    text = pattern_endpoints.sub(gemini_endpoints, text)

# 4. Remove defunct generate_symptom_analysis function
pattern_func = re.compile(r'def generate_symptom_analysis.*?return responses\.get\(language, responses\["en"\]\)\n', re.DOTALL)
text = pattern_func.sub('', text)

text = text.replace('- POST /analyze/symptoms', '- POST /chat/message\\n    - POST /chat/reset')

with codecs.open('main.py', 'w', 'utf-8') as f:
    f.write(text)

print("Patch applied to main.py successfully!")
