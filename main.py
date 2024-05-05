import cv2
import os
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import clip
import torchvision.transforms as transforms
import numpy as np
import json

class od_llm:

    consolidated_data_list = []
    consolidated_dict = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    name = 'SHREYAS'

    # Function to extract frames from video
    def extract_frames(self, video_path, output_folder):
        # Open the video file
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Iterate through frames and save as images
        while success:
            frame_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_path, image)     # save frame as JPEG file
            success, image = vidcap.read()
            print(f'Frame {count} saved: {frame_path}')
            count += 1

    def preprocess_image(self,image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_tensor = od_llm.preprocess(image_pil).unsqueeze(0).to(od_llm.device)
        return image_tensor
    
    def get_embedding(self, image):


        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            image_features = od_llm.model.encode_image(image_tensor)
            
        return image_features.cpu().numpy()


    def get_objects(self, frame):
        prefix = "./obj_det_test_vid_frames"
        label_box_dict = {}
        print(frame)
        image = Image.open(os.path.join(prefix, frame))

        # you can specify the revision tag if you don't want the timm dependency
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            # print(
            #         f"Detected {model.config.id2label[label.item()]} with confidence "
            #         f"{round(score.item(), 3)} at location {box}"
            # )
            # print(model.config.id2label[label.item()], box)
            label_box_dict[str(box)] = model.config.id2label[label.item()]
        
        embedding = self.get_embedding(image)
        print("id: ", frame)
        print("EMBEDDING: ", embedding[0])
        print("metadata: ", label_box_dict)
        # od_llm.consolidated_data_list.append([frame, embedding[0], label_box_dict])
        od_llm.consolidated_data_list.append({
            'id': frame,
            'values': embedding[0],
            'metadata': label_box_dict
        })
        

        print("\n")



    def create_metadata(self, frames_folder):

        for frame in os.listdir(frames_folder):
            self.get_objects(frame)
    
    def json_dump(self):
        # print(od_llm.consolidated_data_list)
        # for each in od_llm.consolidated_data_list:
        #     od_llm.consolidated_dict.append({
        #         'id': each[0],
        #         'values': each[1],
        #         'metadata': each[2]
        #     })

        for item in od_llm.consolidated_data_list:
            item["values"] = item["values"].tolist()

        # Save data to a JSON file
        json_file_path = 'obj_det_test_vid_data.json'
        with open(json_file_path, 'w') as json_file:
            json.dump(od_llm.consolidated_data_list, json_file)

if __name__ == "__main__":

    # Example usage
    video_path = "obj_det_test_vid.mp4"  # Replace "your_video.mp4" with the path to your video file
    output_folder = "obj_det_test_vid_frames"       # Output folder to save frames

    det = od_llm()
    # det.extract_frames(video_path, output_folder)
    det.create_metadata(output_folder)
    det.json_dump()
