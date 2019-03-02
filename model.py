import torch.nn as nn
import torchvision

class Model(nn.Module):
  def __init__(self, num_cats):
    super(Model, self).__init__()
    # download resnet34 pretrained
    self.model = torchvision.models.resnet34(pretrained=True)
    # freeze
    for param in self.model.parameters():
      param.requires_grad = False                                      # Evita reentrenar la red neuronal, así utiliza el preentreno de Resnet34
    # add new fc layer
    self.model.fc = nn.Linear(self.model.fc.in_features, num_cats)     # Cambia solamente la ultima capa para adaptarse a nuestro problema (21 categorias)
  def forward(self, x):                                                # Función que se ejecuta, recibe la imagen de entrada
    x = self.model(x)
    return x
