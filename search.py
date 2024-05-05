import json
import openai
import torch
import clip
from PIL import Image
import cv2
from transformers import AutoProcessor, LlavaForConditionalGeneration
import ast

class Search:
    openai.api_key = 'sk-proj-iwODL7jgP51elXlQ0MYyT3BlbkFJvt15Sf3yLG83eMmmQIwW'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    meta_filter_results = []

    def get_sentence_object(self, sentence):

        # Initialize OpenAI API client

        # client = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo-instruct"
        # )
        content = "what is the object in the sentence: {}. answer in one key. give me just the object of the sentence.".format(sentence)
        # Define the conversation prompt
        conversation = [
            {"role": "system", "content": "You are an English teacher who knows perfect grammar."},
            {"role": "user", "content": content}
        ]

        # Generate a chat completion response (using text-babbage-001)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=conversation
        )

        # Print the generated completion
        # print(completion.choices[0].message['content'])
        return completion.choices[0].message['content']


    def get_probable_embeddings(self, sentence):

        sentence_object = self.get_sentence_object(sentence)
        print("Looking for: ", sentence_object)
        # Load JSON data from file
        with open('obj_det_test_vid_data.json', 'r') as json_file:
            data = json.load(json_file)

        # Loop through each object in the JSON data and display its content
        for obj in data:
            # print("ID:", obj['id'])
            # print("Values:", obj['values'])
            # print("Metadata:")
            # print(obj['metadata'].items())
            for key, value in obj['metadata'].items():
                if value == sentence_object:
                    # print(f"{key}: {value}")
                    # print(obj['values'])
                    embedding_tensor = torch.tensor(obj['values']).reshape(1, -1)
                    Search.meta_filter_results.append([obj['id'], embedding_tensor, obj['metadata']])
                    print("\n")


    def get_description(self, frame):
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        prompt = "USER: <image>\ndescribe what is around the the laptop mouse. ASSISTANT:"
        url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        image = Image.open('./obj_det_test_vid_frames/' + frame)

        inputs = processor(text=prompt, images=image, return_tensors="pt")

        # Generate
        generate_ids = model.generate(**inputs, max_new_tokens=15)
        description = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(description)

    def draw_bounding_boxes(self, frame, bounding_boxes):
        # Load the image
        image = cv2.imread('./obj_det_test_vid_frames/' + frame)
        
        # Draw bounding boxes
        for bbox in bounding_boxes:
            # Parse the bounding box coordinates
            xmin, ymin, xmax, ymax = map(int, bbox)
            
            # Draw the bounding box rectangle
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green color, thickness=2
        
        # Display the image with bounding boxes
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_cosine_similarity_scores(self, prompt):
        text_input = clip.tokenize([prompt]).to(Search.device)
        text_features = Search.model.encode_text(text_input)
        sentence_embedding = text_features[0].detach().cpu().numpy()
        sentence_embedding_tensor = torch.tensor(sentence_embedding)
        # print(sentence_embedding)
        max_score, best_frame, metadata = 0, '', []
        for each in Search.meta_filter_results:
            print(each[0])
            cosine_similarity = torch.nn.functional.cosine_similarity(sentence_embedding_tensor, each[1]).item()
            print(f"Cosine similarity score: {cosine_similarity}")
            if cosine_similarity > max_score:
                max_score = cosine_similarity
                best_frame = each[0]
                metadata = each[2]
            print("\n")
        
        # print(best_frame, max_score, next(iter(metadata.keys())), type(list(next(iter(metadata.keys())))) )
        bounding_boxes = [ast.literal_eval(box) for box in metadata.keys()]
        self.draw_bounding_boxes(best_frame, bounding_boxes )
        # image = Image.open('./obj_det_test_vid_frames/' + best_frame)
        # # self.get_description(best_frame)
        # image.show()

        

if __name__ == "__main__":
    search = Search()

    prompt = "I cant find my backpack. where is it?"
    search.get_probable_embeddings(prompt)
    search.get_cosine_similarity_scores(prompt)