from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel

from app.services.text_service import TextGenerator

router = APIRouter(tags=["text"])
text_service = TextGenerator()

class TextGenerationRequest(BaseModel):
    sign_words: List[str]
    emotion: str

class TextGenerationResponse(BaseModel):
    generated_text: str

@router.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    try:
        generated_text = text_service.generate_sentence(request.sign_words, request.emotion)
        
        return TextGenerationResponse(generated_text=generated_text)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="문장 생성 중 오류가 발생했습니다.") 