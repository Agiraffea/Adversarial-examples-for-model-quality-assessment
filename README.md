![image](https://github.com/Agiraffea/model_test_program/blob/main/6cf154166a758891fe8e70064c6269a.png)
# Model_Test_Program
   Our program can be used to evaluate the reliability of image classification models. We generate adversarial samples by adding cloud layers and Gaussian blur to visible light images. For SAR images, we introduce high-scattering noise points in the background area and modify pixel patch in the object area to generate adversarial samples. By introducing these interferences, we can assess the robustness of the models.
 
---
## How to use our code

After downloading our code, you simply need to open the following file to use it.

    main.py


Or you can install with `pip`.
```python
pip install soimt
```
### Load your model and dataset

In this section, you can input the path of the model and the data you want to test into the code.
```python
dataloder_test_path = 'adversarial_attack_for_optical/dataset/test'
model_path_for_optic = 'adversarial_attack_for_optical/checkpoint/ResNet18.pth'
```
Dataloder_test_path is the path of the dataset you want to test.
Model_path_for_optic is the path of the model you want to test.

---
If you need to preprocess your model and dataset, you can perform the following steps. If no preprocessing is required, please proceed to the next section.
```python
model = torch.load('adversarial_attack_for_optical/checkpoint/ResNet18.pth')
model = model['net']
model_SAR = MobileNetV2()
model_SAR.load_state_dict(torch.load("adversarial_attack_for_SAR/point_attack/MSTAR_MobileNetV2_0709.model"))
```    
    
You can load your model parameters or perform other operations in the module as shown in the example above.
```python
transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Resize((256, 256)),
   transforms.Normalize((0.4842, 0.4901, 0.4505), (0.1735, 0.1635, 0.1555))
])
```
You can preprocess your dataset as shown in the example above.

If you want to test SAR dataset, please change the path below.
```python
test_data_dir = 'adversarial_attack_for_SAR/sticker_attack/test128'
```
---
### Generate adversarial samples you want
#### Alterations to optical remote sensing images

Adversarial method  | Original image | Altered image
:-------------: | ------------- | -------------
Add cloud  | ![](https://github.com/Agiraffea/model_test_program/blob/main/result%20example/optical%20result/cloud/golfcourse.12.png) | ![](https://github.com/Agiraffea/model_test_program/blob/main/result%20example/optical%20result/cloud/img1_true_golfcourse_pred_tenniscourt.png)
Add blur  | ![](https://github.com/Agiraffea/model_test_program/blob/main/result%20example/optical%20result/blur/tenniscourt.11.png) | ![](https://github.com/Agiraffea/model_test_program/blob/main/result%20example/optical%20result/blur/img1_true_tenniscourt_pred_baseballdiamond.png)

It is preferable that the images in your dataset have a size of 256 px × 256 px.
Here is the demonstration of the attack process.
```
----------The NO.1 image attack succeeded----------
Success Rate: 100.00% (1/1)
.
.
.
----------The NO.89 image attack failed----------
Success Rate: 32.58% (29/89)
----------The NO.90 image attack failed----------
Success Rate: 32.22% (29/90)
For [untargeted attack] on [ResNet18] network model, the final success rate is: 32.22%
```
The success rate of the attack is 32.22%, and the model precision is 67.78%.
Here is the result line chart.
![](https://github.com/Agiraffea/model_test_program/blob/main/result%20example/result%20picture_optic.png)

#### Alterations to SAR images

Adversarial method  | Original image | Altered image
:-------------: | ------------- | -------------
High-scattering points  | ![](https://github.com/Agiraffea/model_test_program/blob/main/result%20example/SAR%20result/point%20attack/HB14932.jpeg) | ![](https://github.com/Agiraffea/model_test_program/blob/main/result%20example/SAR%20result/point%20attack/HB14932_1.jpeg)
Add pixel patch  | ![](https://github.com/Agiraffea/model_test_program/blob/main/result%20example/SAR%20result/patch%20attack/HB14941.jpeg) | ![](https://github.com/Agiraffea/model_test_program/blob/main/result%20example/SAR%20result/patch%20attack/HB14941_1.jpg)

It is preferable that the images in your dataset have a size of 128 px × 128 px.
Here is the result chart.
![](https://github.com/Agiraffea/model_test_program/blob/main/result%20example/result%20picture_SAR.png)


