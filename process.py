import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import clip
import sys
import os
import json


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

num_frames = 10
frame_interval = 1

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    image_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    return image_tensor

videos_folder = '/Users/shreyasjoshi/Documents/contribs/rough_work/object_detection_data'
video_embeddings = []
for video in os.listdir(videos_folder):
  if ".mp4" in video:
    print(video)
    video_path = os.path.join(videos_folder,video)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    frame_embeddings = []
    for i in tqdm(range(num_frames)):
        frame_pos = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            break

        image_tensor = preprocess_image(frame)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            frame_embeddings.append(image_features.cpu().numpy())

    video_embedding = np.mean(frame_embeddings, axis=0)
    # print(video_embedding[0])
    # Convert to numpy array
    embedding_array = np.array(video_embedding[0])
    # Save to .npy file
    np.save('embedding_vector.npy', embedding_array)

    video_embeddings.append([video,video_embedding])
    cap.release()
    print("Video embedding shape:", video_embedding.shape)

data = []

for each in video_embeddings:
  data.append({
      'id': each[0],
      'values': each[1]
  })


for item in data:
    item["values"] = item["values"].tolist()

# Save data to a JSON file
json_file_path = 'data.json'
with open(json_file_path, 'w') as json_file:
    json.dump(data, json_file)

print("Data saved to JSON file.")