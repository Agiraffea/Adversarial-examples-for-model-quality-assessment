# Model_Test_Program
   Our project can be used to evaluate the reliability of image classification models. We generate adversarial samples by adding cloud layers and Gaussian blur to visible light images. For SAR images, we introduce high-scattering noise points in the background area and modify pixel blocks in the object area to generate adversarial samples. By introducing these interferences, we can assess the robustness of the models.
 
tags: `adversarial attack`,`SAR`,
 
code: `python`
 
---
## How to use our code

After downloading our code, you simply need to open the following file to use it.

    main.py

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
#### Alterations to optical remote sensing images.

Adversarial method  | Original image | Altered image
------------- | ------------- | -------------
Add cloud  | ![][https://github.com/Agiraffea/model_test_program/blob/main/result%20example/optical%20result/cloud/golfcourse.12.tif] | ![][https://github.com/Agiraffea/model_test_program/blob/main/result%20example/optical%20result/cloud/img1_true_golfcourse_pred_tenniscourt.png]
Add blur  | ![][https://github.com/Agiraffea/model_test_program/blob/main/result%20example/optical%20result/blur/tenniscourt.11.tif] | ![][https://github.com/Agiraffea/model_test_program/blob/main/result%20example/optical%20result/blur/img1_true_tenniscourt_pred_baseballdiamond.png]



