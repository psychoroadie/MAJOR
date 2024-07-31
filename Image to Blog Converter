import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from roboflow import Roboflow
import openai

class ImageUploader:
    def __init__(self, master):
        self.master = master
        self.master.title("Image to Blog Generator")

        self.master.grid_columnconfigure(0, weight=1, minsize=300)
        self.master.grid_columnconfigure(1, weight=1, minsize=300)

        title_label = tk.Label(self.master, text="Image to Blog Generator", font=("Helvetica", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.image_label = tk.Label(self.master)
        self.image_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.name_label = tk.Label(self.master, text="DESCRIPTION :", justify="left", wraplength=400)
        self.name_label.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        self.upload_button = tk.Button(self.master, text="Upload", command=self.upload_image)
        self.upload_button.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        self.pause_button = tk.Button(self.master, text="Pause", command=self.pause_resume)
        self.pause_button.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

        self.pause_state = False
        self.current_name = ""
        self.display_index = 0

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                
                import torch
                import torchvision.models as models
                import torchvision.transforms as transforms
                from PIL import Image
                from torchvision.models import inception_v3
                from torchvision.models.detection import FasterRCNN
                from torchvision.models.detection.rpn import AnchorGenerator
                from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
                import torch
                from PIL import Image

                # Define a new instance of the Inception V3 model
                model = inception_v3(pretrained=False, aux_logits=False)

                # Modify the number of output classes in the current model to match the number of output classes in the saved state_dict
                num_classes = 18
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

                # Load the saved state_dict into the model
                model_state_dict = torch.load('inception_v3_trained_subject.pth')
                model_dict = model.state_dict()

                # Set the model to evaluation mode
                model.eval()

                # Define the transform to preprocess the input image
                transform = transforms.Compose([
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                ])

            
                image_path = file_path
                image = Image.open(image_path)
                image = transform(image)
                tensor = image.unsqueeze(0)  
                detected_objects=[]
                with torch.no_grad():
                    out = model(tensor)
                probabilities = torch.nn.functional.softmax(out[0], dim=0)
                with open("C:/Users/91703/OneDrive/Desktop/Major Project/imagenet_classes_subject.txt", "r") as f:
                    categories = [s.strip() for s in f.readlines()]
                top5_prob, top5_catid = torch.topk(probabilities,3)
                out = [categories[top5_catid[i]] for i in range(3)]
                for i in range (3):
                    detected_objects.append(out[i])

                


                from roboflow import Roboflow
                rf = Roboflow(api_key="")
                project = rf.workspace("major-ndv6h").project("blogify-vision")
                version = project.version(1).model
                #dataset = version.download("tensorflow")
                 
                detected_objects = []
                import cv2


                image = cv2.imread(file_path)


                results = version.predict(image)


                predictions = results.json()
                for prediction in predictions["predictions"]:
                    detected_objects.append(prediction["class"])
                print(detected_objects)


               



                # VGG19
                from tensorflow.keras.applications.vgg19 import VGG19, decode_predictions, preprocess_input
                from tensorflow.keras.preprocessing import image

                model = VGG19(weights='imagenet')
                #detected_objects = []

                img = image.load_img(file_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                predictions = model.predict(x)
                output = decode_predictions(predictions, top=4)[0]

                for item in output:
                    detected_objects.append(item[1])
                print(detected_objects)    


                # YOLO MODEL
                rf = Roboflow(api_key="")
                project = rf.workspace("").project("image-to-blog")
                Model = project.version(1).model
                image = cv2.imread(file_path)
                results = Model.predict(image)
                predictions = results.json()

                for prediction in predictions["predictions"]:
                    detected_objects.append(prediction["class"])
                print(detected_objects)    


                from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
                import torch
                from PIL import Image

                model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

                feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
                tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                max_length = 16
                num_beams = 4
                gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

                #detected_objects = []

                def predict_step(image_paths):
                    images = []
                
                    for image_path in image_paths:
                        i_image = Image.open(image_path)
                        if i_image.mode != "RGB":
                            i_image = i_image.convert(mode="RGB")

                        images.append(i_image)

                    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
                    pixel_values = pixel_values.to(device)

                    output_ids = model.generate(pixel_values, **gen_kwargs)

                    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    preds = [pred.strip() for pred in preds]
                    return preds

                pred = predict_step([file_path])
                #print(pred)
                detected_objects += pred
                #print(detected_objects)



                # OPENAI API
                openai.organization = ""
                openai.api_key = ""

                def generate_blog(keywords):
                    response = openai.completions.create(
                        model="gpt-3.5-turbo-instruct",
                        prompt='Write a 300 words blog which mainly describes about the scenic view using the following keywords. ' + ','.join(keywords),
                        max_tokens=2500
                    )
                    return response.choices[0].text

                blog = generate_blog(detected_objects)

                # Display the image and blog
                image = Image.open(file_path)
                image = image.resize((300, 300), Image.LANCZOS)  # or Image.BICUBIC
                self.image_tk = ImageTk.PhotoImage(image)
                self.image_label.config(image=self.image_tk)
                self.image_label.image = self.image_tk

                self.current_name = blog
                self.display_index = 0
                self.display_image_name()
            except Exception as e:
                print("Error:", e)

    def display_image_name(self):
        self.name_label.config(text="DESCRIPTION : ")
        self.master.update()
        if not self.pause_state:
            self.display_chars()

    def display_chars(self):
        if self.display_index < len(self.current_name):
            char = self.current_name[self.display_index]
            self.name_label.config(text=self.name_label.cget("text") + char)
            self.display_index += 1
            if not self.pause_state:
                self.master.after(50, self.display_chars)

    def pause_resume(self):
        if self.pause_state:
            self.pause_button.config(text="Pause")
            self.pause_state = False
            self.display_chars()
        else:
            self.pause_button.config(text="Resume")
            self.pause_state = True

    def clear_image_name(self):
        self.name_label.config(text="Image Name: ")

def main():
    root = tk.Tk()
    app = ImageUploader(root)
    root.mainloop()

if __name__ == "__main__":
    main()
