# 开发日期  2024/8/2
import torch
import  torchvision
from PIL import  Image
from torch import nn

image_path="test_picture/plane1.jpg"
image=Image.open(image_path)
image=image.convert('RGB')
print(image)

transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])
image=transform(image)
print(image.shape)


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x=self.model(x)
        return x


model=torch.load("model_save/mynet_29.pth",map_location=torch.device("cpu"))
print(model)
image=torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output=model(image)
print(output)
print(output.argmax(1))