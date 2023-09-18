# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List
from tools.Dataset import Dataset
import subprocess

app = FastAPI()

# Create an instance of the Dataset class to handle dataset loading
dataset = None

# Define your data models using Pydantic
class Hashtags(BaseModel):
   hashtags: List[str]

# Feature One: Upload a txt file containing hashtags and create a dataset
@app.post("/upload/")
async def create_dataset(file: UploadFile = File(...)):
    global dataset

    # Implement the logic to create a dataset from the uploaded file and hashtags
    # Example: You can save the file, preprocess it, and create the dataset

    try:
        # Save the uploaded file
        with open('stop_words/hashtags.txt', 'wb') as f:
            file_data = file.file.read()
            f.write(file_data)

        # Call your create_dataset.py script using subprocess
        subprocess.run(['python', 'tools/create_dataset.py', '--hashtags', 'stop_words/local_hashtags.json'])

        # Load the newly created dataset using your Dataset class
        # dataset = Dataset("path_to_your_data_folder")
        return {"message": "Dataset created successfully"}
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

# ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
