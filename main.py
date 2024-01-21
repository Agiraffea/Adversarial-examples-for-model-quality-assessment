import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision import transforms
import visdom
############################################################################################################
#This part is the path associated with the optical image
##########
#The path for the dataset to be tested
##########
dataloder_test_path = 'adversarial_attack_for_optical/dataset/test'
##########
#The path for the model to be tested
##########
model_path_for_optic = 'adversarial_attack_for_optical/checkpoint/ResNet18.pth'

############################################################################################################
### Define your own network:
class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        num_features = self.mobilenet.classifier[1].in_features

        # 修改输入通道数2
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # 修改预训练模型的权重参数
        pretrained_weights = self.mobilenet.features[0][0].weight
        new_weights = torch.sum(pretrained_weights, dim=1, keepdim=True)
        self.mobilenet.features[0][0].weight = nn.Parameter(new_weights)

        # 修改分类器部分
        self.mobilenet.classifier[1] = nn.Linear(num_features, 10)

    def forward(self, x):
        x = self.mobilenet(x)
        return x
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=False)
        num_features = self.densenet.classifier.in_features
        self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.densenet.classifier = nn.Linear(num_features, 10)

    def forward(self, x):
        x = self.densenet(x)
        return x
class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2)
        self.ReLU = nn.ReLU()
        self.c2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.s2 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.s3 = nn.MaxPool2d(2)
        self.c4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(4608, 2048)
        self.f7 = nn.Linear(2048, 2048)
        self.f8 = nn.Linear(2048, 1000)

        self.f9 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.s3(x)
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s5(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = F.dropout(x, p=0.5)
        x = self.f7(x)
        x = F.dropout(x, p=0.5)
        x = self.f8(x)
        x = F.dropout(x, p=0.5)

        x = self.f9(x)
        x = F.dropout(x, p=0.5)
        return x
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.relu_1=nn.ReLU()
        self.max_pool2d_1=nn.MaxPool2d(kernel_size=2)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.relu_2 = nn.ReLU()
        self.max_pool2d_2 = nn.MaxPool2d(kernel_size=2)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.relu_3 = nn.ReLU()
        self.max_pool2d_3 = nn.MaxPool2d(kernel_size=2)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu_4 = nn.ReLU()
        self.flatten=nn.Flatten()
        self.fc_1 = nn.Linear(in_features=12 * 12 * 64, out_features=200)
        self.relu_5 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=200, out_features=200)
        self.relu_6 = nn.ReLU()
        self.fc_3 = nn.Linear(in_features=200, out_features=10)

    def forward(self, x):
        x=self.conv_1(x)
        x=self.relu_1(x)
        x=self.max_pool2d_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.max_pool2d_2(x)
        x = self.conv_3(x)
        x = self.relu_3(x)
        x = self.max_pool2d_3(x)
        x = self.conv_4(x)
        x = self.relu_4(x)
        x = self.flatten(x)
        x=self.fc_1(x)
        x=self.relu_5(x)
        x = self.fc_2(x)
        x = self.relu_6(x)
        x=self.fc_3(x)
        return x

### Load your own model:
model = torch.load('adversarial_attack_for_optical/checkpoint/ResNet18.pth')
model = model['net']
model_SAR = MobileNetV2()
model_SAR.load_state_dict(torch.load("adversarial_attack_for_SAR/point_attack/MSTAR_MobileNetV2_0709.model"))
###Define your own transform：
###Please make sure that transform[-1] is transforms.Normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize((0.4842, 0.4901, 0.4505), (0.1735, 0.1635, 0.1555))
])
############################################################################################################
#This part is the path associated with the SAR image

#point attack method:
#The path for the dataset to be tested
folder_path = './adversarial_attack_for_SAR/point_attack/test1000/'
#The path where the attacked image will be saved
saved_path = './adversarial_attack_for_SAR/point_attack/test_add_point/'
#The path where the image mask will be saved
segment_path = './adversarial_attack_for_SAR/point_attack/seg1000/'


#stick attack method:
#The path for the dataset to be tested
test_data_dir = 'adversarial_attack_for_SAR/sticker_attack/test128'
############################################################################################################

############################################################################################################

print("If you want to test optical image model, please enter 'optic'")
print("If you want to test SAR image model, please enter 'SAR'")
class_input = input('Enter your command(optic / SAR)：')
if class_input == 'optic':
    print("If you want to generate clouds for images, please enter 'cloud'")
    print("If you want to blur the images, please enter 'blur'")
    user_input = input('Enter the image you want to generate(cloud / blur):')
    if user_input == 'cloud':
        from adversarial_attack_for_optical import adversarial_cloud_attack
        adversarial_cloud_attack.load_data_cloud(dataloder_test_path, model, transform)

    if user_input == 'blur':
        from adversarial_attack_for_optical import adversarial_blur_attack

        adversarial_blur_attack.load_data_blur(dataloder_test_path, model, transform)


if class_input == 'SAR':
    print("If you want to generate an image with one highlighted noise point, please enter 'one'")
    print("If you want to generate an image with three highlighted noise points, please enter 'three'")
    print("If you want to generate an image with five highlighted noise points, please enter 'five'")
    print("If you want to generate an image with patch noise, please enter 'patch'")
    user_input = input('Enter the image you want to generate(one / three / five / patch):')
    if user_input == 'one':
        from adversarial_attack_for_SAR.point_attack import yym_seg_1_point_20231004
        print("----------generating disturbed dataset----------")
        yym_seg_1_point_20231004.one_point(folder_path, saved_path, segment_path, model_SAR)
        print("----------generating disturbed dataset complete----------")
        yym_seg_1_point_20231004.start_attack(model_SAR)

    if user_input == 'three':
        from adversarial_attack_for_SAR.point_attack import yym_seg_3_point_20231005
        print("----------generating disturbed dataset----------")
        yym_seg_3_point_20231005.three_points(folder_path, saved_path, segment_path, model_SAR)
        print("----------generating disturbed dataset complete----------")
        yym_seg_3_point_20231005.start_attack(model_SAR)

    if user_input == 'five':
        from adversarial_attack_for_SAR.point_attack import yym_seg_5_point_20231006
        print("----------generating disturbed dataset----------")
        yym_seg_5_point_20231006.five_points(folder_path, saved_path, segment_path, model_SAR)
        print("----------generating disturbed dataset complete----------")
        yym_seg_5_point_20231006.start_attack(model_SAR)

    if user_input == 'patch':
        from adversarial_attack_for_SAR.sticker_attack import yym_attack_five_models_20230919
        print("----------generating disturbed dataset----------")
        yym_attack_five_models_20230919.attack_model(test_data_dir, model_SAR)


