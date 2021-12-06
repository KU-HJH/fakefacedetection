import subprocess
import os
import random
from tqdm import tqdm

root = 'dataset/style2'

all_images = os.listdir(root)

total = len(all_images)

for x in ['train', 'val']:
    os.makedirs(os.path.join(root, x), exist_ok=True)

val_len = total // 5

val_images = random.sample(all_images, val_len)

print('Val images: ', len(val_images))
for f in tqdm(val_images):
    if not os.path.isdir(os.path.join(root, f)):
        command = ['mv', os.path.join(root, f), os.path.join(root, 'val', f)]
        subprocess.check_output(command)

rest_images = os.listdir(root)
print('Train images: ', len(rest_images))
for f in tqdm(rest_images):
    if not os.path.isdir(os.path.join(root, f)):
        command = ['mv', os.path.join(root, f), os.path.join(root, 'train', f)]
        subprocess.check_output(command)
