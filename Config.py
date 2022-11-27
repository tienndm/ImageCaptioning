class Config():
    embedSize = 512
    hiddenSize = 512
    numLayers = 1
    numEpochs = 100
    learning_rate= 3e-4
    tensorboardDir = "runs/flickr"
    imageDir = r'D:\ImageCaptioning\DataFlick8k\Train\Images'
    captionDir = r'D:\ImageCaptioning\DataFlick8k\Train\captions.txt'
    numWorker = 2