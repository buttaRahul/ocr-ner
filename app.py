from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import os
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

load_dotenv()

endpoint = os.getenv('ENDPOINT')
key = os.getenv("API_KEY")

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

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def clean_text(text):
    tokens = text.split()
    cleaned_tokens = [word for word in tokens if word.lower() not in stop_words and word not in punctuation]
    return " ".join(cleaned_tokens)

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

def merge_subwords(ner_results):
    merged_results = []
    for entity in ner_results:
        if entity['word'].startswith("##"):
            if merged_results and merged_results[-1]['entity_group'] == entity['entity_group']:
                merged_results[-1]['word'] += entity['word'][2:]
            else:
                entity['word'] = entity['word'][2:]
                merged_results.append(entity)
        else:
            merged_results.append(entity)
    return merged_results

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def extract_text_from_pdf_file(file: UploadFile = File(...)):
    try:
        extracted_text = extract_text_from_pdf(file.file)
        cleaned_text = clean_text(extracted_text)
        return {"extracted_text": cleaned_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error extracting text: " + str(e))

@app.post("/ner")
async def analyze_text_with_ner(text: str = Form(...)):
    try:
        cleaned_text = clean_text(text)
        ner_results = ner(cleaned_text)
        ner_results = merge_subwords(ner_results)
        categorized_entities = {}
        for entity in ner_results:
            entity_type = entity['entity_group']
            if entity_type not in categorized_entities:
                categorized_entities[entity_type] = set()
            categorized_entities[entity_type].add(entity['word'])

        categorized_entities = {k: list(v) for k, v in categorized_entities.items()}
        return categorized_entities
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error performing NER: " + str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
