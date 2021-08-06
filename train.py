import os
import pandas as pd
import numpy as np
import albumentations
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import engine
from model import get_model
import dataset
import matplotlib.pyplot as plt
#import torchvision

import warnings

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    # location of train.csv and train_png folder
    # with all the png images
    csv_path = '/data/Train.csv'
    csv_test_path =  '/data/Test.csv'
    data_path = "/data/train/"
    data_path_test = "/data/Test/"

    # cuda/cpu device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # let's train for 10 epochs
    epochs = 10
    # load the dataframe
    df = pd.read_csv(csv_path)
    df_test = pd.read_csv(csv_test_path)
    df_test["LABEL"] = -1
    # fetch all image ids
    images = df.ID.values.tolist()
    images_test = df_test.ID.values.tolist()
    # a list with image locations
    images = [
        os.path.join(data_path, i + ".png") for i in images
    ]
    images_test = [
        os.path.join(data_path_test, i + ".png") for i in images_test
    ]
    # binary targets numpy array
    targets = df.LABEL.values
    targets_test = df_test.LABEL.values
    # fetch out model, we will try both pretrained
    # and non-pretrained weights
    model = get_model(pretrained=True)
    # move model to device
    model.to(device)
    # mean and std values of RGB channels for imagenet dataset
    # we use these pre-calculated values when we use weights
    # from imagenet.
    # when we do not use imagenet weights, we use the mean and
    # standard deviation values of the original dataset
    # please note that this is a separate calculation
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # albumentations is an image augmentation library
    # that allows you do to many different types of image
    # augmentations. here, i am using only normalization
    # notice always_apply=True. we always want to apply
    # normalization
    aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            )
        ]
    )
    # instead of using kfold, i am using train_test_split
    # with a fixed random state
    train_images, valid_images, train_targets, valid_targets = train_test_split(images, targets, stratify=targets, random_state=42    )
    # fetch the ClassificationDataset class
    train_dataset = dataset.ClassificationDataset(
        image_paths=train_images,targets=train_targets,resize=(227, 227),augmentations=aug, )
    
    # from classification dataset class
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4
    )

    #dataiter = iter(train_loader)
    #id, LABEL = dataiter.next()
    
    #imshow(torchvision.utils.make_grid(id))
    # print labels
    #print(' '.join('%5s' % classes[LABEL[j]] for j in range(16)))
    # same for validation data
    valid_dataset = dataset.ClassificationDataset(
        image_paths=valid_images,
        targets=valid_targets,
        resize=(227, 227),
        augmentations=aug,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    test_dataset = dataset.ClassificationDataset(
        image_paths=images_test,
        targets=targets_test,
        resize=(227, 227),
        augmentations=aug,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4
    )
    # simple Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    # train and print auc score for all epochs
    for epoch in range(epochs):
        engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_targets = engine.evaluate(valid_loader, model, device=device )
        roc_auc = metrics.roc_auc_score(valid_targets, predictions)
        print( f"Epoch={epoch}, Valid ROC AUC={roc_auc}" )

    PATH = 'C:\\Users\\MziyandaP\\PycharmProjects\\XrayRecognition\\data\\xray_net.pth'
    torch.save(model.state_dict(),PATH)
    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(PATH))
   
    final_targets = []
    final_outputs = []
    final_inputs  = []
    with torch.no_grad():
        for data in test_loader:
            inputs = data["ID"]
            targets = data["LABEL"]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            output = model(inputs)
            output = torch.sigmoid(output)
            output = torch.round(output)
            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()

            # extend the original list
            
            final_targets.extend(targets)
            final_outputs.extend(output)
    df_output = pd.DataFrame()
    df_output["ID"] = np.array(df_test.ID.values.tolist())
    df_output["LABEL"] = np.array(final_outputs)
    df_output.to_csv("/data/Submit.csv",index=False)





        






   

