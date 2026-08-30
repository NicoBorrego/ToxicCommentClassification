from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import classifier
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Definition of entry data structure, in this case a comment, so string
class InputText(BaseModel):
    text: str

cls = classifier.Classifier()

@app.post("/predict")
def predict(entry: InputText):
    return cls.predict(entry.text)