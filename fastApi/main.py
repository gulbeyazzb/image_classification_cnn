from fastapi import FastAPI, File, UploadFile 
from pydantic import BaseModel # Fonksiyonlara gelen parametreleri kontrol etmek için
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from fastapi.responses import  HTMLResponse
from PIL import Image

app = FastAPI()


@app.get("/")
async def root():
    with open("C:/Users/Hp/Documents/GitHub/image_classification_cnn/fastApi/index.html", encoding="utf-8") as file:
        content = file.read()
    return HTMLResponse(content)

class model_input(BaseModel):
    Unnamed: int
    Age: int
    Gender: int
    B0I: int
    Chol: float
    TG: float
    HDL: float
    LDL: float
    Cr: float
    BUN : float

diabetes_model=load_model('models/model.h5')

@app.post("/predict_diabetes")
def predict(input_parameters : model_input ):
     data = pd.DataFrame([input_parameters.model_dump()])
     sc = StandardScaler()
    
     sc.fit(data)
     data = sc.transform(data)
     
     prediction = diabetes_model.predict(data)
     print(prediction)

     if prediction[0][0] == 1.0:
         return {"message": "The person is Diabetic"}
     else:
         return {"message": "The person is not Diabetic"}


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    load_cnn_model = load_model(r'C:\Users\Hp\Documents\GitHub\image_classification_cnn\fastApi\models\model_cnn.h5')
    with open("uploaded_img.jpg", "wb") as f:
        f.write(await file.read())
        
    test_image = Image.open("uploaded_img.jpg")
    test_image = test_image.resize((240, 240))
    
    test_image = np.array(test_image)
    test_image = np.expand_dims(test_image, axis=0) / 255.0
    
    prediction = load_cnn_model.predict(test_image)
    # predicted_class_index = np.argmax(prediction[0][0])

    if prediction[0][0] <= 0.5:
        return 'Jerry'
    else:
         return 'Tom'



