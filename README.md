# HackAI

This project is a collection of AI tools. We focused on AI model compression and optimization using smaller AI models, quantization, knowledge distillation, and pruning. Theses models were used on the topic of fire classification and detection.

Around this network compression we built a edge application using a Jetson Xavier nx card. This application was used to detect fire in the wild and send a notification to the user using telegram. To access the application, the user needs to go through a face recognition system.

## Models

Before choosing our final model, we tried a lot of models. We used a lot of different models and applied some finetuning. Overall in terms of performance, the model that gave us the best results was the Squeezenet. The Squeezenet model is a good model for image classification offering a good balance between accuracy and efficiency.

Using another dataset, we tried using the Vision Transformer model. The Vision Transformer model offers exceptionnal accuracy, but it is also very slow.

| Architecture       | Hyperparameters           | Precision (Acc %) | Loss   | Model Size (MB) |
| ------------------ | ------------------------- | ----------------- | ------ | --------------- |
| ResNet152          | Epoch: 10, Batch size: 16 | 96.5              | 0.1118 | 89.7            |
| ResNet152          | Epoch: 25, Batch size: 16 | 89.93             | 0.65   | 223.8           |
| ResNet152          | Epoch: 35, Batch size: 16 | 89.26             | 0.65   | 223             |
| ResNet152          | Epoch: 4, Batch size: 16  | 85.38             | 0.70   | 223             |
| SqueezeNet         | Epoch: 35, Batch size: 16 | 95.45             | 0.11   | -               |
| SqueezeNet         | Epoch: 25, Batch size: 16 | 93.97             | 0.1753 | 2.8             |
| VGG16              | Epoch: 10, Batch size: 16 | 90.06             | 0.65   | 520             |
| Wide-resnet        | Epoch: 10, Batch size: 16 | 90.52             | 0.6479 | 259.4           |
| Vision Transformer | Epoch: 4, Batch size: 16  | 1                 | 0,0068 | 343             |

## Optimization

Once our model is chosen, we can optimize it. We used a lot of different methods to optimize the model.

### Quantization

We applied quantization to all of our final models. The quantization process allows us to reduce the size of the model by 2x (depending on number of bits chosen) while keeping the same level of accuracy. This is a great way to reduce the size of the model and the amount of memory it takes on the device.

### Knowledge Distillation

With our more elaborated model (Resnet152), we can use knowledge distillation to improve the model. Knowledge distillation is a process where a smaller model is trained to be similar to the original model. The smaller model used was Mobilenet. However, it did not improved the model's accuracy.

| Architecture | Hyperparameters           | Precision (Acc %) | Loss   | Model Size (MB) |
| ------------ | ------------------------- | ----------------- | ------ | --------------- |
| Mobilenet    | Epoch: 10, Batch size: 16 | 91.28             | 0.6426 | 9.4             |
| Mobilenet    | Epoch: 20, Batch size: 32 | 89.01             | 0.6578 | 9.4             |

### Pruning

We tried pruning on the Squeezenet model. However, due to incompatibility between the model and the pruning tool, we could not apply pruning. One solution to rewrite the pruning script provided for the Hackathon.

## Folders

The repository contains the following folders:

- 1_face_recognition: Contains the face recognition system.
- 2_image_classification: Contains the image classification system.
- 3_fire_detection: Contains the fire detection system.
- 4_telegram_bot: Contains the telegram bot.
- HackIA24_Input: Contains the starting code from the Hackathon.

## How to use

The notebooks are made to be used in Google Colab. A requirements.txt is provided to install the dependencies. To use the telegram bot, you need to replace the TOKEN in the 4_telegram_bot/main.py file with your token.
