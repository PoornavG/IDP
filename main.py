# main.py
import os
import time
import io
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddlewares # <--- OLD/INCORRECT
from fastapi.middleware.cors import CORSMiddleware # <--- CORRECT

# Import your pipeline functions here
from your_pipeline import process_image, process_batch_with_reports

# You might want to define a custom route class to always return JSONResponse
# This is an advanced option and might not be strictly necessary, but can be cleaner
class CustomJSONResponseRoute(APIRoute):
    def get_route_handler(self):
        original_route_handler = super().get_route_handler()
        async def custom_route_handler(request):
            response = await original_route_handler(request)
            if isinstance(response, dict):
                return JSONResponse(response)
            return response
        return custom_route_handler

app = FastAPI(
    # route_class=CustomJSONResponseRoute # Uncomment if you want this global behavior
)

# Allow your frontend origin
app.add_middleware(
    # CORSMiddlewares, # <--- OLD/INCORRECT
    CORSMiddleware, # <--- CORRECT
    allow_origins=["http://localhost:8080"], # Or your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

@app.post("/process/single")
async def process_single(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Images only.")
    contents = await file.read()
    image_stream = io.BytesIO(contents)

    try:
        result = await process_image(image_stream, MEDIA_DIR)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")


@app.post("/process/batch")
async def process_batch(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    image_streams_with_names = []
    for file in files:
        contents = await file.read()
        image_streams_with_names.append((io.BytesIO(contents), file.filename))

    try:
        batch_output = await process_batch_with_reports(image_streams_with_names, MEDIA_DIR)
        return JSONResponse(batch_output)
        
    except Exception as e:
        print(f"Error during batch processing: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {e}")


@app.get("/media/{path:path}")
async def media(path: str):
    media_path = MEDIA_DIR / path
    if not media_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(media_path)