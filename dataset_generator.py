dataset_filename = "Gleb_dataset.pkl"

import json
import os
from PIL import Image #pip install --upgrade Pillow
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pickle

images_folder = f"dataset/stage2/"
labels = {}
csv_names = os.listdir("dataset/stage4")
transform = transforms.Compose([transforms.ToTensor()])
image_list = []
labels_list = []
strings = '0123456789abcdefghijklmnpqrstuvwxyz'
label_mapping = {strings[i]:i for i in range(35)}
label_mapping['o'] = 0
print(label_mapping)


for i in csv_names:
    with open(f"dataset/stage4/{i}", 'r') as data:
        temp_labels = json.load(data)
        labels.update(temp_labels)



for img_name, label in labels.items():
    image_path = os.path.join(images_folder, img_name)
    image = Image.open(image_path)
    image_tensor = transform(image)
    image_list.append(image_tensor)
    if label!='':
        labels_list.append(label_mapping[label])

custom_dataset = {'data': torch.stack(image_list), 'targets': torch.tensor(labels_list)}


with open(dataset_filename, 'wb') as file:
    pickle.dump(custom_dataset, file)