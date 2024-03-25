from fastapi import FastAPI, File, UploadFile
from ..models.zhaomodel import ZhaoModel
from torch import Tensor

app = FastAPI()

@app.post("/query")
async def query_model(search_request: UploadFile = File(...)):
    """Send an image to the model, and get the 10 closest images back.
    \nSample usage (Python and HTML):\n
    requests.post(url=f"{url}/query", files={"search_request": open(image_name, "rb")})
    \n\nOR\n\n
    <body>\n
    <form action='/query' enctype='multipart/form-data' method='post'>\n
    <input name='file' type='file'>\n
    <input type='submit'>\n
    </form>\n
    </body>
    \nTODO: Implement this function"""
    
    # Load the image (try...except...finally block)
    #   - Save the image to a temporary file
    #   - Load the image from the temporary file
    # Convert the image to a tensor
    # Load the model
    # Query the model
    # Return the result in a format that can be easily displayed in the frontend.
    pass