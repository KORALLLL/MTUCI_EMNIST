import json
import os

images_folder = f"dataset/stage2/"
labels_folder =f"dataset/stage4/"

labels = {}

csv_names = os.listdir(labels_folder)

# with open(labels_folder, 'r') as data:
#     labels = json.load(data)



# with open('dataset/final_labels/my_label.csv', 'w') as file:
#     json.dump(new_labels, file)