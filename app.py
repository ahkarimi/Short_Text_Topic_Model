# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List
from tools.Dataset import Dataset
import subprocess
import os

app = FastAPI()

DATA_FOLDER = "data"

# Ensure the upload folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)

# Create an instance of the Dataset class to handle dataset loading
dataset = None

# Define your data models using Pydantic
class Hashtags(BaseModel):
   hashtags: List[str]

# Feature One: Upload a txt file containing hashtags and scrape a dataset
@app.post("/scrape_data/")
async def scrape_dataset(file: UploadFile = File(...)):
    global dataset

    try:
        # Save the uploaded file
        with open('data/hashtags.txt', 'wb') as f:
            file_data = file.file.read()
            f.write(file_data)

        subprocess.run(['python', 'tools/create_dataset.py', '--hashtags', 'data/local_hashtags.json'])

        return {"message": "Dataset scraped successfully"}
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))


# Feature One: Upload a txt file containing hashtags and create a dataset
@app.post("/upload_data/")
async def create_dataset(file: UploadFile = File(...)):
    global dataset

    try:

        # Determine the file format (TSV) based on the file extension
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension != "tsv":
            return HTTPException(status_code=400, detail="Unsupported file format (Please upload a tsv file)")

        file_path = os.path.join(DATA_FOLDER, 'data.tsv')
    
        # Save the uploaded file
        with open(file_path, 'wb') as f:
            f.write(file.file.read())

        # Load the newly created dataset using your Dataset class
        dataset = Dataset(DATA_FOLDER)
        return {"message": "Dataset created successfully"}
    
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
