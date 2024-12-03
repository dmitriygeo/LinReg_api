from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
from fastapi.encoders import jsonable_encoder
import io

app = FastAPI()


class Item(BaseModel):
    year: int
    km_driven: int
    mileage: float
    engine: int
    max_power: float
    seats: float


class Items(BaseModel):
    objects: List[Item]


with open("model.pkl", 'rb') as model_file:
    loaded_model = pickle.load(model_file)

def pydantic_model_to_df(model_instance):
    return pd.DataFrame([jsonable_encoder(model_instance)])

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df_instance = pydantic_model_to_df(item)
    prediction = loaded_model.predict(df_instance).tolist()[0]
    return prediction


@app.post("/predict_items")
def predict_items(items: UploadFile) -> List[float]:
    content = items.file.read().decode("utf-8")
    stream = io.StringIO(content)
    df = pd.read_csv(stream, delimiter=',')

    df_columns = df.columns.tolist()
    input_data = df[df_columns].values
    predictions = loaded_model.predict(input_data)
    df['predicted_price'] = predictions

    stream_output = io.StringIO()
    df.to_csv(stream_output, index=False, sep=';')

    response = StreamingResponse(iter([stream_output.getvalue()]),
                                 media_type="text/csv"
                                 )
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response
