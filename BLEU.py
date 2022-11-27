from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
import torch
from Predict import predict
import pandas as pd
import os
import spacy
from tqdm import tqdm
    
# spacy_eng = spacy.load("en_core_web_sm")

root = r'D:\ImageCaptioning\DataFlick8k\Val'

captionFile = os.path.join(root,'captions.txt')

df = pd.read_csv(captionFile,sep='|')

nameImg = '667626_18933d713e.jpg'

imageData = df['image'].to_list()

captionData = df['caption'].to_list()

imgDir = os.path.join(root,'Images')

model = torch.load(r'D:\ImageCaptioning\models\26-11-2022\LastModel.pt',map_location='cuda:0')

sumBLEU = 0
idx = 0

for nameImg in tqdm(os.listdir(imgDir)):
    img = os.path.join(imgDir,nameImg)
    result = predict(img,model)
    # print(f'Model predict: {result}')
    for i in range(5):
        caption = captionData[imageData.index(nameImg)+ i]
        # print(caption)
        bleu = sentence_bleu(caption,result,weights=(1.0,0,0,0))
        sumBLEU += bleu
        idx += 1
print(f'Avg BLEU1: {sumBLEU/idx}')
# actual = 'A girl is stretched out in shallow water'
# bleu = sentence_bleu(actual,result)
# print(bleu)
