from torchvision.models import vgg16
from torch import nn

base_model = vgg16(pretrained=False)
layers = list(base_model.features)[:-1]
model = nn.Sequential(*layers)
print(model)
