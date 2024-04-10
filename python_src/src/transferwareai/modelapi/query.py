import logging
import time
from typing import Annotated

import torchvision
from fastapi import FastAPI, File, HTTPException, Depends
import torch
from pydantic import BaseModel

from transferwareai.config import settings
from transferwareai.modelapi.model import (
    initialize_model,
    get_model,
    get_api,
)
from transferwareai.models.adt import ImageMatch, Model
from transferwareai.tccapi.api_cache import ApiCache
from fastapi.responses import FileResponse

app = FastAPI()


@app.on_event("startup")
def startup():
    initialize_model()


@app.post("/query", response_model=list[ImageMatch])
async def query_model(
    file: Annotated[bytes, File()], model: Annotated[Model, Depends(get_model)]
):
    """Send an image to the model, and get the 10 closest images back."""

    start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

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

    end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    logging.debug(f"Query took {(end - start) / 1e6}ms.")

    return top_matches


@app.get("/pattern/image/{id}")
async def get_image_for_id(id: int, api: Annotated[ApiCache, Depends(get_api)]):
    """Gets the main image for a pattern ID."""
    p = api.get_image_file_path_for_tag(id, "pattern")
    return FileResponse(p)


class Metadata(BaseModel):
    pattern_id: int
    pattern_name: str
    tcc_url: str


@app.get("/pattern/{id}", response_model=Metadata)
async def get_data_for_pattern(id: int, api: Annotated[ApiCache, Depends(get_api)]):
    """Gets metadata for a pattern ID."""
    name = api.get_name_for_pattern_id(id)
    url = api.get_tcc_url_for_pattern_id(id)

    return Metadata(pattern_id=id, pattern_name=name, tcc_url=url)
