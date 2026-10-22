import time, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_TRAIN="/workspace/data/cifar10/train"
MEAN=(0.4914,0.4822,0.4465)
STD=(0.2470,0.2435,0.2616)

for B in [512,1024,2048]:
    for aug in ["simple","auto"]:
        if aug=="simple":
            tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
        else:
            tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),transforms.ToTensor(),transforms.Normalize(MEAN,STD),transforms.RandomErasing(p=0.25)])
        ds=datasets.ImageFolder(DATA_TRAIN, transform=tf)
        for nw in [4,8]:
            loader=DataLoader(ds,batch_size=B,shuffle=True,num_workers=nw,pin_memory=True,persistent_workers=True,prefetch_factor=2,drop_last=True)
            # warmup
            t0=time.time()
            for i,(x,y) in enumerate(loader):
                x=x.cuda(non_blocking=True)
                if i>=5: break
            # timed epoch
            t0=time.time()
            n=0
            for x,y in loader:
                x=x.cuda(non_blocking=True)
                n+=1
            dt=time.time()-t0
            print(f"B={B} aug={aug} nw={nw} steps={n} time={dt:.2f}s per_step={dt/n*1000:.1f}ms")
