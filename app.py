from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi import Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import os
from dotenv import load_dotenv
load_dotenv()

endpoint = os.getenv('ENDPOINT')
# print(endpoint)
key = os.getenv("API_KEY")
print(key)




app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

origins = [
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

document_analysis_client = DocumentAnalysisClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)

ner = pipeline(
    'ner',
    model="D:\\download\\saved_model-20240515T173042Z-001\\saved_model",
    aggregation_strategy='simple',
    device=0
)

def extract_text_from_pdf(pdf_file):
    poller = document_analysis_client.begin_analyze_document(
        "prebuilt-read", document=pdf_file
    )
    result = poller.result()

    extracted_text = ""
    for page in result.pages:
        for line in page.lines:
            extracted_text += line.content + " "

    return extracted_text

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def extract_text_from_pdf_file(file: UploadFile = File(...)):
    try:
        extracted_text = extract_text_from_pdf(file.file)
        return {"extracted_text": extracted_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error extracting text: " + str(e))

@app.post("/ner")
async def analyze_text_with_ner(text: str = Form(...)):
    try:
        ner_results = ner(text)
        categorized_entities = {}
        for entity in ner_results:
            entity_type = entity['entity_group']
            if entity_type not in categorized_entities:
                categorized_entities[entity_type] = []
            categorized_entities[entity_type].append(entity['word'])
        return categorized_entities
    except Exception as e:
        raise HTTPException(status_code=5000, detail="Error performing NER: " + str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
