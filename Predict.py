import torch
from torchvision import transforms
from Model import CNNtoRNN
from PIL import Image
from GetLoader import getLoader
from Config import Config
import cv2

def predict(image,model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")
    transform = transforms.Compose([transforms.Resize((299, 299)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    _, dataset = getLoader(root_folder = Config.imageDir,
                           annotation_file = Config.captionDir,
                           transform = transform,
                           numWorkers = Config.numWorker,)
    model.eval()
    processImg = transform(Image.open(image)).unsqueeze(0)
    result = " ".join(model.captionImages(processImg.to(device),dataset.vocab))
    result = result.replace('<SOS> ','')
    result = result.replace('<EOS>','')
    print("Result: " + result)




