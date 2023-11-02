from fastapi import FastAPI
from pydantic import BaseModel
from cnn_mfcc import *

app = FastAPI()

def model_predict(model, mfcc_features):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    model.load_weights("best_weights/Weights")

    model.predict(mfcc_features)
    classes = np.argmax(predictions, axis = 1)

class StockIn(BaseModel):
    path: str


class StockOut(StockIn):
    label: int

@app.get("/ping")
def pong():
    return {"ping": "pong!"}


