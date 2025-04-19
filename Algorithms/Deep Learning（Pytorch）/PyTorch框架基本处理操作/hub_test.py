# GITHUB:https://github.com/pytorch/hub
# 模型：https://pytorch.org/hub/research-models
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights, deeplabv3_resnet101

import os
os.environ['GITHUB_TOKEN'] = 'GITHUB_TOKEN'  # github令牌，替换为自己的，否则报错urllib.error.HTTPError: HTTP Error 401: Unauthorized

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', weights=True)  # deprecated
# model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)  # 官方推荐写法
model.eval()  # 将这些层设置到预测模式,如果在预测的时候忘记使用model.eval()，会导致不一致的预测结果
print(torch.hub.list('pytorch/vision:v0.10.0'))

# Download an example image from the pytorch website
# import urllib

# url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
# try:
#     urllib.URLopener().retrieve(url, filename)
# except:
#     urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)

input_image = Image.open("./data/dog.jpg")
input_image = input_image.convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

plt.imshow(r)
plt.show()

# all above from github
