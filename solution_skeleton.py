from gmm import GMMSimple, train_model
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


app = FastAPI()
# Set up the static files (JS, CSS, images, etc.) directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    with open("./index.html") as f:
        html = f.read()

    return HTMLResponse(content=html, status_code=200)
