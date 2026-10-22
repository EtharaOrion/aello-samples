import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader

device=torch.device("cuda")
mean=(0.4914,0.4822,0.4465); std=(0.2470,0.2435,0.2616)
tf_train=T.Compose([T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(mean,std)])
tf_test=T.Compose([T.ToTensor(),T.Normalize(mean,std)])
train_set=ImageFolder("/workspace/data/cifar10/train",transform=tf_train)
test_set=ImageFolder("/workspace/data/cifar10/test",transform=tf_test)
train_loader=DataLoader(train_set,batch_size=128,shuffle=True,num_workers=2)
test_loader=DataLoader(test_set,batch_size=512,shuffle=False,num_workers=2)

# Simple ResNet
class BasicBlock(nn.Module):
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(ci,co,3,stride,1,bias=False)
        self.bn1=nn.BatchNorm2d(co)
        self.conv2=nn.Conv2d(co,co,3,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(co)
        self.shortcut=nn.Sequential()
        if stride!=1 or ci!=co:
            self.shortcut=nn.Sequential(nn.Conv2d(ci,co,1,stride,bias=False),nn.BatchNorm2d(co))
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,3,1,1,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.layer1=self._make(64,64,2,1)
        self.layer2=self._make(64,128,2,2)
        self.layer3=self._make(128,256,2,2)
        self.layer4=self._make(256,512,2,2)
        self.fc=nn.Linear(512,10)
    def _make(self,ci,co,n,stride):
        layers=[BasicBlock(ci,co,stride)]
        for _ in range(1,n): layers.append(BasicBlock(co,co))
        return nn.Sequential(*layers)
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.layer1(out); out=self.layer2(out); out=self.layer3(out); out=self.layer4(out)
        out=F.adaptive_avg_pool2d(out,1).flatten(1)
        return self.fc(out)

model=ResNet().to(device)
print(sum(p.numel() for p in model.parameters()))
opt=torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
crit=nn.CrossEntropyLoss()
scaler=torch.amp.GradScaler('cuda')
for epoch in range(3):
    model.train()
    tot=0; correct=0; loss_sum=0
    for x,y in train_loader:
        x,y=x.to(device),y.to(device)
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            out=model(x)
            loss=crit(out,y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loss_sum+=loss.item()*x.size(0)
        correct+=(out.argmax(1)==y).sum().item()
        tot+=x.size(0)
    print(f"Ep{epoch} loss {loss_sum/tot:.3f} acc {correct/tot:.3f}")
    model.eval()
    correct=tot=0
    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device),y.to(device)
            out=model(x)
            correct+=(out.argmax(1)==y).sum().item()
            tot+=y.size(0)
    print(f"  test {correct/tot:.4f}")
