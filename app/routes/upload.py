from fastapi import APIRouter, UploadFile
import shutil

from app.utils.parser import parse_pdf
from app.utils.chunking import chunk_text
from app.services.embed import get_embedding
from app.db.chroma import add_docs

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile):
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = parse_pdf(file_path)
    print('text', text)
    chunks = chunk_text(text)
    print('chunks', chunks)
    embeddings = get_embedding(chunks)
    print('embeddings', embeddings)

    add_docs(chunks, embeddings)

    return {"message": "File processed successfully"}