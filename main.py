from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from self_rag import EnhancedSelfRAG
import logging
import os
import uuid
import shutil
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS

import urllib
from datetime import datetime
import traceback
import json

# Environment setup
# Add Azure OpenAI configuration
os.environ['AZURE_OPENAI_API_KEY'] = '8b1d436e85d1452bbcbfd5905921efa6'


# Initialize FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files setup
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components
self_rag = EnhancedSelfRAG()
ALLOWED_EXTENSIONS = {'pdf'}
generated_strategies_cache = {}
source_tracking = {
    'documents': {},
    'urls': {},
}

# Pydantic models
class URLInput(BaseModel):
    url: str
    sector: str = "general"

# Pydantic models
class StrategyInput(BaseModel):
    sector: str
    parameters: Dict[str, Any] = {}

    class Config:
        json_schema_extra = {
            "example": {
                "sector": "credit_card",
                "parameters": {
                    "credit_score": 750,
                    "annual_income": 75000,
                    "monthly_spending": 3000,
                    "preferred_rewards": "travel",  # Options: travel, cashback, points, business
                    "card_type": "rewards"  # Options: rewards, business, secured, student
                }
            }
        }

    

class MoreInfoInput(BaseModel):
    strategy: Dict[str, Any]
    urls: List[str]

class SourceInput(BaseModel):
    source: str


@app.post("/upload_document")
async def upload_document(document: UploadFile = File(...)):
    try:
        if not document.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Please upload PDF documents only")

        document_id = str(uuid.uuid4())
        filename = f"{document_id}_{document.filename}"
        permanent_filepath = os.path.join(self_rag.pdf_dir, filename)

        # Create directory and save file directly
        os.makedirs(self_rag.pdf_dir, exist_ok=True)
        with open(permanent_filepath, "wb") as buffer:
            shutil.copyfileobj(document.file, buffer)

        return {
            "status": "success",
            "document_id": document_id,
            "message": "Document successfully uploaded",
            "details": {
                "filename": document.filename,
                "is_indexed": False
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/process_url")
async def process_url(input_data: URLInput):
    try:
        return {
            "status": "success",
            "message": "URL received",
            "details": {
                "url": input_data.url,
                "sector": input_data.sector
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_strategies")
async def generate_strategies(input_data: StrategyInput):
    try:
        sector = input_data.sector
        parameters = input_data.parameters or {} 

        if sector not in generated_strategies_cache:
            generated_strategies_cache[sector] = set()
        
        # Store and validate parameters first
        validated_params = self_rag.store_sector_parameters(sector, input_data.parameters)
        
        

        result = self_rag.query(
            question=f"Generate strategies for {sector} sector",
            sector=sector
        )
        

        # Rest of your existing endpoint code...
        
        return {
            "strategies": result.get('strategies', []),
            "metadata": {
                "sector": sector,
                "parameters_used": validated_params,
                "total_generated": len(generated_strategies_cache[sector])
            }
        }

    except Exception as e:
        logging.error(f"Strategy generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/view_pdf_page")
async def view_pdf_page(path: str, page: int = 1):
    try:
        # Extract both filename and page reference from path
        parts = path.split(':')
        pdf_filename = parts[0].strip()
        
        # Handle UUID prefix in filename
        pdf_files = [f for f in os.listdir(self_rag.pdf_dir) if f.lower().endswith('.pdf')]
        target_pdf = next((pdf for pdf in pdf_files if pdf_filename.lower() in pdf.lower()), None)
        
        if not target_pdf:
            # Search by UUID if direct filename match fails
            target_pdf = next((pdf for pdf in pdf_files if pdf.startswith(pdf_filename)), None)
            
        if not target_pdf:
            raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_filename}")

        # Ensure static directory exists
        static_pdf_dir = os.path.join("static", "pdfs")
        os.makedirs(static_pdf_dir, exist_ok=True)
        
        # Copy file to static directory with original name preserved
        source_path = os.path.join(self_rag.pdf_dir, target_pdf)
        static_pdf_path = os.path.join(static_pdf_dir, target_pdf)
        shutil.copy2(source_path, static_pdf_path)
        
        # Return URL with page anchor
        return {
            "url": f"/static/pdfs/{target_pdf}#page={page}",
            "page": page,
            "filename": target_pdf,
            "success": True,
            "absolute_url": True,
            "full_path": static_pdf_path
        }

    except Exception as e:
      
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/get_more_info")
async def get_more_info(input_data: MoreInfoInput):
    try:
        result = self_rag.get_more_info(
            strategy=input_data.strategy,
            urls=input_data.urls
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
