import cell_cropping
import json
import matplotlib
import numpy as np
import os
import torch
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import transforms as t
from matplotlib import pyplot as plt
from PIL import Image


# editable parameters, set up to fit local file structure
##################################################
img_path = 'Test_Images/'             # folder where images are
model_path = 'models/'                # folder where model is stored
model_name = 'model_97.pth'           # trained model name
save_path = 'visuals/'                # location to save figures
defect_dir = 'defect_percentages/'    # location to save defect percentage jsons
defect_per = False                    # turn on if you want to see defect percentages
##################################################
################ model parameters ################
pre_model = 'deeplabv3_resnet50'      # backbone model was trained on
num_classes = 5                       # number of classes model trained on
threshold = .52                       # threshold for defect interpretation
aux_loss = True                       # loss type model trained with
##################################################

filelist = os.listdir(img_path)
# print(filelist)

# softmax layer for defect interpretation
softmax = torch.nn.Softmax(dim=0)

# this section loads in the weights of an already trained model
model = torchvision.models.segmentation.__dict__[pre_model](aux_loss=aux_loss,
                                                            pretrained=True)

# changes last layer for output of appropriate class number
if pre_model == 'deeplabv3_resnet50' or pre_model == 'deeplabv3_resnet101':
    model.classifier = DeepLabHead(2048, num_classes)
else:
    num_ftrs_aux = model.aux_classifier[4].in_channels
    num_ftrs = model.classifier[4].in_channels
    model.aux_classifier[4] = torch.nn.Conv2d(num_ftrs_aux, num_classes, kernel_size=1)
    model.classifier[4] = torch.nn.Conv2d(num_ftrs, num_classes, kernel_size=1)

# model = model.cuda()
checkpoint = torch.load(model_path+model_name, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

# transforms to put images through the model
trans = t.Compose([t.ToTensor(), t.Normalize(mean=.5, std=.2)])

# create custom colormap for image visualizations [Black, Red, Blue, Purple, Orange]
cmaplist = [(0.001462, 0.000466, 0.013866, 1.0),
            (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0),
            (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0),
            (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0),
            (1.0, 0.4980392156862745, 0.0, 1.0)]

# create the new map
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom', cmaplist, len(cmaplist))

i = 0
# loops through every image in folder
while i < len(filelist):
    # opens up and preps image, runs through model (RGB to benefit from pretrained model)
    if os.path.isdir(img_path+filelist[i]):
        i += 1
        continue
    im = Image.open(img_path+filelist[i]).convert('RGB')
    # meant to capture module images and crop them
    if im.size[0] > 2000:
        cell_cropping.CellCropComplete(img_path+filelist[i], i, NumCells_x=12, NumCells_y=6)
        split = os.listdir(img_path + 'Cell_Images' + str(i) + '/')
        split = ['Cell_Images' + str(i) + '/' + s for s in split]
        filelist.extend(split)
        i += 1
        continue
    print(filelist)
    img = trans(im).unsqueeze(0)
    output = model(img)['out']

    # threshold to determine defect vs. non-defect instead of softmax (custom for this model)
    soft = softmax(output[0])
    nodef = soft[0]
    nodef[nodef < threshold] = -1
    nodef[nodef >= threshold] = 0
    nodef = nodef.type(torch.int)
    def_idx = soft[1:].argmax(0).type(torch.int)
    def_idx = def_idx + 1
    nodef[nodef == -1] = def_idx[nodef == -1]

    if defect_per:
        # name is for saving json with defect percentage
        name = 'cell' + str(i)

        # counts stats of pixels/defect percentages
        output_pix = torch.count_nonzero(nodef)
        total_pix = torch.numel(nodef.detach())

        output_defect_percent = torch.div(output_pix.type(torch.float), total_pix)

        crack_portion = torch.div(torch.count_nonzero(nodef == 1), total_pix)
        contact_portion = torch.div(torch.count_nonzero(nodef == 2), total_pix)
        interconnect_portion = torch.div(torch.count_nonzero(nodef == 3), total_pix)

        # creates json to save defect percentage per class category
        defect_percentages = {'crack': round(float(crack_portion), 7), 'contact': round(float(contact_portion), 7),
                              'interconnect': round(float(interconnect_portion), 7)}

        with open(defect_dir + name + '.json', 'w') as fp:
            json.dump(defect_percentages, fp)

    orig_img = (img * .2) + .5
    nodef = np.ma.masked_where(nodef == 0, nodef)

    # plots the original image next to prediction (with defect percentage)
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img[0][0].numpy(), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title('original image')
    plt.subplot(1, 2, 2)
    plt.imshow(orig_img[0][0], cmap='gray', vmin=0, vmax=1)
    plt.imshow(nodef, cmap=cmap, vmin=0, vmax=4, alpha=.3)
    plt.title('image + prediction')
    plt.tick_params(axis='both', labelsize=0, length=0)
    if defect_per:
        plt.xlabel("Defect Percentage: " + str(output_defect_percent.numpy().round(5)))

    # plt.savefig(save_path + str(i) + '.png') # comment back in to save figures
    plt.show()
    plt.clf()
    i += 1

