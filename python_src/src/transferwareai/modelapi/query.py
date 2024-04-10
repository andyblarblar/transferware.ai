from typing import Annotated

import torchvision
from fastapi import FastAPI, File, HTTPException, Depends
import torch

from transferwareai.config import settings
from transferwareai.modelapi.model import initialize_model, get_model
from transferwareai.models.adt import ImageMatch, Model

app = FastAPI()


@app.on_event("startup")
def startup():
    initialize_model()


@app.post("/query", response_model=list[ImageMatch])
async def query_model(
    file: Annotated[bytes, File()], model: Annotated[Model, Depends(get_model)]
):
    """Send an image to the model, and get the 10 closest images back."""

    # Parse image into tensor
    try:
        raw_tensor = torch.frombuffer(file, dtype=torch.uint8)
        img = torchvision.io.decode_image(raw_tensor, torchvision.io.ImageReadMode.RGB)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail="Only jpg or png files are supported"
        )

    # Query model
    top_matches = model.query(img, top_k=settings.query.top_k)

    return top_matches
