from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
from model import Model
from PIL import Image
import numpy as np
import torch

app = Flask(__name__)
CORS(app)
api = Api(app)

cats = [
    'airplane',
    'runway',
    'golfcourse',
    'agricultural',
    'mediumresidential',
    'mobilehomepark',
    'river',
    'tenniscourt',
    'parkinglot',
    'overpass',
    'freeway',
    'denseresidential',
    'chaparral',
    'harbor',
    'storagetanks',
    'buildings',
    'sparseresidential',
    'forest',
    'intersection',
    'baseballdiamond',
    'beach'
]

# load model
net = torch.load('./model.pth')
net.eval()     # red en modo evaluacion   -----------

class Predict(Resource):
    def post(self):       #  metodo utilizado por WEB flask

        # get image from request
        img = request.files.get('file')      # recibe la imagen
        img.save('./'+img.filename)          # guarda la imagen

        # load image
        img = Image.open(img.filename).convert('RGB').resize((224,224))    # la abre y redimensiona la imagen a 224 x 224
        img = np.array(img)                                                # la tranforma a un array de numpy
        img = torch.FloatTensor(img.transpose((2,0,1)) / 255)              # Transformacion de la imagen a channel y colores de RGB a GRB
                                                                          # 2 canales, cambia el canal

        # get predictions
        pred = net(img.unsqueeze(0)).squeeze()                            # pasa un batch de solo una imagen, añade un campo nuevo con el numero de imagenes
        pred_lab = torch.argmax(pred).item()                              # argmax devuelve el indice de la prediccion más alta
        prediction = cats[pred_lab]                                       # busca el indice de la prediccion en la lista de categorias

        return {'prediction': prediction}

api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(port=8000, debug=True)
