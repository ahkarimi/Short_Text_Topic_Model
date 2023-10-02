# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List
from tools.Dataset import Dataset
from models import LDA, CTM, BTM, NMF
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

class TrainRequest(BaseModel):
    models: List[str]
    hyperparameters: dict

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



# Feature Two: Train models with selected model names
@app.post("/train/")
async def train_models(train_request: TrainRequest, dataset_id: str = Form(...)):
    global dataset

    if dataset is None:
        return HTTPException(status_code=400, detail="Dataset not created yet")

    # Implement the logic to train models with the selected names using the dataset
    training_results = {}

    for model_name in train_request.models:
        if model_name == "LDA":
            # Example: Train an LDA model
            lda = LDA.LDA(num_topics=train_request.hyperparameters.get("num_topics", 10),
                      iterations=train_request.hyperparameters.get("iterations", 5))
            lda_result = lda.train_model(dataset, hyperparams=None, top_words=10)
            training_results["LDA"] = lda_result

        elif model_name == "CTM":
            # Example: Train a CTM model
            ctm = CTM.CTM(num_topics=train_request.hyperparameters.get("num_topics", 10),
                      num_epochs=train_request.hyperparameters.get("epoch", 5),
                      bert_model=train_request.hyperparameters.get("bert_model", "m3hrdadfi/bert-zwnj-wnli-mean-tokens"))
            ctm_result = ctm.train_model(dataset)
            training_results["CTM"] = ctm_result

        elif model_name == "NMF":
            # Example: Train an NMF model
            nmf = NMF.NMF(num_topics=train_request.hyperparameters.get("num_topics", 10))
            nmf_result = nmf.train_model(dataset)
            training_results["NMF"] = nmf_result
        
        elif model_name == "BTM":
            # Example: Train an NMF model
            btm = BTM.BTM(num_topics=train_request.hyperparameters.get("num_topics", 10),
                      num_iterations=train_request.hyperparameters.get("iterations", 5))
            btm_result = btm.train_model(dataset)
            training_results["BTM"] = btm_result

    return training_results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
