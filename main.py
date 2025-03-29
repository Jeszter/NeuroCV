import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import fitz
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    print(f"Gemini initialization error: {str(e)}")

system_prompt = """
You are a professional HR consultant with 10 years of experience in recruitment. 
Your task is to thoroughly analyze resumes and provide detailed recommendations for improvement.

Analysis format:

1. **Overall resume assessment** (on a 5-point scale)
   - Strengths
   - Critical weaknesses

2. **Detailed section breakdown**:
   - Contact information: {analysis}
   - Work experience: {analysis}
   - Education: {analysis}
   - Skills: {analysis}
   - Additional sections: {analysis}

3. **Specific improvement recommendations**:
   - What to add
   - What to rephrase
   - What to remove
   - Optimal structure for this field

4. **Potential red flags** for HR:
   - Gaps in employment history
   - Date inconsistencies
   - Overly generic wording

5. **Adaptation recommendations**:
   - Which job positions to adapt for
   - Which keywords to add for ATS
   - Which achievements to highlight

Example of a good resume for comparison:

**Work Experience**:
- Company X (2020-2023)
  * Position: Senior Developer
  * Achievements: 
    - Optimized processes, reducing task completion time by 30%
    - Led a team of 5 developers
    - Implemented CI/CD system, reducing deployment time by 40%

Your analysis should be specific, professional, and contain practical recommendations.
"""

def extract_text_from_pdf(file: UploadFile) -> str:
    try:
        contents = file.file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF reading error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        return JSONResponse({"error": "File must be in PDF format"}, status_code=400)

    try:
        text = extract_text_from_pdf(file)
        prompt = f"{system_prompt}\n\n--- RESUME FOR ANALYSIS ---\n{text}\n\n--- END OF RESUME ---\n\nPlease provide a detailed analysis according to the specified scheme."

        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 3000,
                "temperature": 0.5,
                "top_p": 0.9
            }
        )

        return JSONResponse({
            "status": "success",
            "analysis": response.text,
            "filename": file.filename
        })

    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"File processing error: {str(e)}"
        }, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)