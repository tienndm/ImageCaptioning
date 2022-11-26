import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from Utils import save_checkpoint,print_examples
from GetLoader import getLoader
import Utils
from Model import CNNtoRNN
from tqdm import tqdm
from Config import Config
from Predict import predict
import os
import cv2

torch.set_num_threads(2)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    transform = transforms.Compose([transforms.Resize((356,356)),
                                    transforms.RandomCrop((299,299)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])
    print(f'Loading dataset ...')
    trainLoader, dataset = getLoader(root_folder = Config.imageDir,
                                      annotation_file = Config.captionDir,
                                      transform = transform,
                                      numWorkers = Config.numWorker,
                                      )
    torch.backends.cudnn.benchmark = True
    loadModel = False
    saveModel = True

    #Hyperparameter
    vocabSize = len(dataset.vocab)

    writer = SummaryWriter("runs/flickr")
    step = 0

    model = CNNtoRNN(Config.embedSize,Config.hiddenSize,vocabSize,Config.numLayers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(),lr=Config.learning_rate)

    if loadModel:
        step = Utils.load_checkpoint()
    
    model.train()
    
    for epoch in range(Config.numEpochs):
        tempLoss = 0
        print(f"{epoch+1}/{Config.numEpochs}")
        if saveModel:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in tqdm(enumerate(trainLoader), total=len(trainLoader)):
            
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
            tempLoss += loss

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
        avrLoss = tempLoss / step
        print(f'Loss: {avrLoss}')
        if (epoch%10==0):
            torch.save(model,f'models/step_{epoch}.pt')
    torch.save(model,f'models/LastModel.pt')
if __name__ == "__main__":
    train()
    # # MODEL_DIR = r'D:\ImageCaptioning\models\9GbDataModel'
    # # for model in os.listdir(MODEL_DIR):
    #     # modelDir = os.path.join(MODEL_DIR,model) 
    #     # print(f'Loading model {model} ...')
    # model = torch.load(r'D:\ImageCaptioning\models\Model.pt',map_location="cuda:0")
    # print(f'Predicting ...')
    # imgDir = r'D:\ImageCaptioning\9GB_Flick\flickr30k_images\flickr30k_images\3662865.jpg'
    # predict(imgDir,model)
    # img = cv2.imread(imgDir)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)

