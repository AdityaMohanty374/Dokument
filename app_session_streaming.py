from fastapi import Request 
from fastapi import Form
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
from fastapi.responses import StreamingResponse
import tempfile
import os
import prompts
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from llm import (
    extract_pdf_text,
    stream_response,
    summarize_text,
    chat,
    clear_chat
)
# session_id -> conversation
conversations = {}
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(
    directory=BASE_DIR / "templates"
)
app = FastAPI(title="Document Summarizer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dokument.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

class SessionRequest(BaseModel):
    session_id: str

def get_messages(session_id: str):
    if session_id not in conversations:
        conversations[session_id] = [prompts.SYSTEM_PROMPT]
    return conversations[session_id]

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index_new2_streaming.html",
        context={
            "request": request
        }
    )

@app.get("/session")
def create_session():
    session_id = str(uuid4())
    conversations[session_id] = [prompts.SYSTEM_PROMPT]
    return {
        "session_id": session_id
    }

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    messages = get_messages(request.session_id)
    chat(messages, request.message)
    def generate():
        answer = ""
        for token in stream_response(messages):
            answer += token
            yield token
        messages.append({
            "role": "assistant",
            "content": answer
        })
    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )


@app.post("/upload")
async def upload_pdf(session_id: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp:
        temp.write(await file.read())
        temp_path = temp.name
    try:
        text = extract_pdf_text(temp_path)
        messages = get_messages(session_id)
        messages.append(
            summarize_text(text)
        )
        def generate():
            answer = ""
            for token in stream_response(messages):
                answer += token
                yield token
            messages.append({
                "role": "assistant",
                "content": answer
            })
        return StreamingResponse(
            generate(),
            media_type="text/plain"
        )
    finally:
        os.remove(temp_path)


@app.post("/clear")
def clear(request: SessionRequest):
    messages = get_messages(request.session_id)
    clear_chat(messages)
    return {
        "message": "Conversation cleared."
    }

