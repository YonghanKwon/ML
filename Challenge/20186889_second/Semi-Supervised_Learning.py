import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import models

import os
from PIL import Image
import argparse




class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = os.listdir(root)
        self.class_to_idx = {c: int(c) for i, c in enumerate(self.classes)}
        self.imgs = []
        for c in self.classes:
            class_dir = os.path.join(root, c)
            for filename in os.listdir(class_dir):
                path = os.path.join(class_dir, filename)
                self.imgs.append((path, self.class_to_idx[c])) 
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target




class CustomDataset_Nolabel(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        ImageList = os.listdir(root)
        self.imgs = []
        for filename in ImageList:
            path = os.path.join(root, filename)
            self.imgs.append(path) 
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img



####################
#If you want to use your own custom model
#Write your code here
####################
class Custom_model(nn.Module):
    def __init__(self):
        super(Custom_model, self).__init__()
        #place your layers
        #CNN, MLP and etc.

    def forward(self, input):
        #place for your model
        #Input: 3* Width * Height
        #Output: Probability of 10 class label
        return predicted_label


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    
####################
#Modify your code here
####################
def model_selection(selection):
    if selection == "resnet":
        model = models.resnet18()
        model.conv1 =  nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        model.layer4 = Identity()
        model.fc = nn.Linear(256, 10)
    elif selection == "vgg":
        model = models.vgg11_bn()
        model.features = nn.Sequential(*list(model.features.children())[:-7])
        model.classifier = nn.Sequential( nn.Linear(in_features=25088, out_features=10, bias=True))
    elif selection == "mobilenet":
        model = models.mobilenet_v2()
        model.classifier = nn.Sequential( nn.Linear(in_features=1280, out_features=10, bias=True))
    elif  selection =='custom':
        model = Custom_model()

    return model

####################
#Modify your code here
####################

####################
#Modify your code here
####################

####################
#Modify your code here
####################

def cotrain(net1,net2, labeled_loader, unlabled_loader, optimizer1_1, optimizer1_2, optimizer2_1, optimizer2_2, criterion):
    #The inputs are as below.
    #{First model, Second model, Loader for labeled dataset with labels, Loader for unlabeled dataset that comes without any labels,
    net1.train()
    net2.train()
    train_loss = 0
    correct = 0
    total = 0
    k = 0.6
    #labeled_training
    for batch_idx, (inputs, targets) in enumerate(labeled_loader):
        inputs, targets = inputs.cuda(), targets.cuda()



####################
#Add your code here
####################
        optimizer1_1.zero_grad()
        outputs1_1=net1(inputs)
        loss1_1 = criterion(outputs1_1,targets)
        loss1_1.backward()
        optimizer1_1.step()

        optimizer2_1.zero_grad()
        outputs2_1=net2(inputs)
        loss2_1 = criterion(outputs2_1,targets)
        loss2_1.backward()
        optimizer2_1.step()

    #unlabeled_training
    for batch_idx, inputs in enumerate(unlabled_loader):

        inputs = inputs.cuda()


####################
#Add your code here
####################
        prediction1 = net1(inputs)
        prediction2 = net2(inputs)
        _, predicted1 = prediction1.max(1)
        _, predicted2 = prediction2.max(1)





        ensemble_prediction=k*prediction1+(1-k)*prediction2
        _,ensemble_predicted=ensemble_prediction.max(1)

        mark=[]

        for ensemble_predicted_idx,ensemble_prediction in zip(ensemble_predicted,ensemble_prediction):
            if ensemble_prediction[ensemble_predicted_idx] < 2:
                mark.append(-1)
            else:
                mark.append(0)

        i=0
        pseudo_label=ensemble_predicted.cuda()
        pseudo_inputs=inputs.cuda()
        for m in mark:
            if m==-1:
                pseudo_inputs=torch.cat([pseudo_inputs[:i],pseudo_inputs[i+1:]],dim=0)
                pseudo_label = torch.cat([pseudo_label[:i], pseudo_label[i+1:]], dim=0)

                i=i-1
            i=i+1


        optimizer1_2.zero_grad()
        outputs1_2=net1(pseudo_inputs)
        loss1_2=criterion(outputs1_2,pseudo_label)
        loss1_2.backward()
        optimizer1_2.step()


        optimizer2_2.zero_grad()
        outputs2_2=net2(pseudo_inputs)
        loss2_2=criterion(outputs2_2,pseudo_label)
        loss2_2.backward()
        optimizer2_2.step()


        #hint :
        #agree = predicted1 == predicted2
        #pseudo_labels from agree


    



def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return 100. * correct / total









if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test',  type=str,  default='False')
    parser.add_argument('--student_abs_path',  type=str,  default='./')
    parser.add_argument('-f')
    args = parser.parse_args()


    batch_size =  20 #Input the number of batch size
    if args.test == 'False':
        train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

        dataset = CustomDataset(root = './data/Semi-Supervised_Learning/labeled', transform = train_transform)
        labeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        dataset = CustomDataset_Nolabel(root = './data/Semi-Supervised_Learning/unlabeled', transform = train_transform)
        unlabeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


        dataset = CustomDataset(root = './data/Semi-Supervised_Learning/val', transform = test_transform)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    else :
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if not os.path.exists(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning')):
        os.makedirs(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning'))



    model_sel_1 =  "resnet"#write your choice of model (e.g., 'vgg')
    model_sel_2 =  "vgg"#write your choice of model (e.g., 'resnet)


    model1 = model_selection(model_sel_1)
    model2 = model_selection(model_sel_2)

    params_1 = sum(p.numel() for p in model1.parameters() if p.requires_grad) / 1e6
    params_2 = sum(p.numel() for p in model2.parameters() if p.requires_grad) / 1e6

    if torch.cuda.is_available():
        model1 = model1.cuda()
    if torch.cuda.is_available():
        model2 = model2.cuda()

    #You may want to write a loader code that loads the model state to continue the learning process
    #Since this learning process may take a while.


    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else :
        criterion = nn.CrossEntropyLoss()


    optimizer1_1 = torch.optim.Adam(model1.parameters(), lr=0.001,betas=(0.9,0.999)) #Optimizer for model 1 in labeled training
    optimizer2_1 = torch.optim.Adam(model2.parameters(), lr=0.001,betas=(0.9,0.999)) #Optimizer for model 2 in labeled training

    optimizer1_2 = torch.optim.Adam(model1.parameters(), lr=0.00002,betas=(0.9,0.999)) #Optimizer for model 1 in unlabeled training
    optimizer2_2 = torch.optim.Adam(model2.parameters(), lr=0.00002,betas=(0.9,0.999)) #Optimizer for model 2 in unlabeled training

    epoch = 200 #Input the number of epochs

    if args.test == 'False':
        assert params_1 < 7.0, "Exceed the limit on the number of model_1 parameters"
        assert params_2 < 7.0, "Exceed the limit on the number of model_2 parameters"

        best_result_1 = 0
        best_result_2 = 0
        for e in range(0, epoch):
            cotrain(model1, model2, labeled_loader, unlabeled_loader, optimizer1_1, optimizer1_2, optimizer2_1, optimizer2_2, criterion)
            tmp_res_1 = test(model1, val_loader)
            # You can change the saving strategy, but you can't change file name/path for each model
            print ("[{}th epoch, model_1] ACC : {}".format(e, tmp_res_1))
            if best_result_1 < tmp_res_1:
                best_result_1 = tmp_res_1
                torch.save(model1.state_dict(),  os.path.join('./logs', 'Semi-Supervised_Learning', 'best_model_1.pt'))

            tmp_res_2 = test(model2, val_loader)
            # You can change save strategy, but you can't change file name/path for each model
            print ("[{}th epoch, model_2] ACC : {}".format(e, tmp_res_2))
            if best_result_2 < tmp_res_2:
                best_result_2 = tmp_res_2
                torch.save(model2.state_dict(),  os.path.join('./logs', 'Semi-Supervised_Learning', 'best_model_2.pt'))
        print('Final performance {} - {}  // {} - {}', best_result_1, params_1, best_result_2, params_2)


    else:
        dataset = CustomDataset(root = '/data/23_1_ML_challenge/Semi-Supervised_Learning/test', transform = test_transform)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model1.load_state_dict(torch.load(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning', 'best_model_1.pt'), map_location=torch.device('cuda')))
        res1 = test(model1, test_loader)

        model2.load_state_dict(torch.load(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning', 'best_model_2.pt'), map_location=torch.device('cuda')))
        res2 = test(model2, test_loader)

        if res1>res2:
            best_res = res1
            best_params = params_1
        else :
            best_res = res2
            best_params = params_2

        print(best_res, ' - ', best_params)