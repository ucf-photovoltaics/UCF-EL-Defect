import copy
import matplotlib
import numpy as np
import os
import torch
import torchvision
from torchvision import transforms as t
from matplotlib import pyplot as plt
from PIL import Image


##################################################
img_path = 'Test_Images/'             # folder where images are
model_path = 'models/'                # folder where model is stored
model_name = 'example_model.pth'      # trained model name
save_path = ''                        # location to save figures
##################################################
pre_model = 'fcn_resnet101'           # backbone model was trained on
classes = 3                           # number of classes model trained on
threshold = .57                       # threshold for defect interpretation
aux_loss = True                       # loss type model trained with
##################################################

filelist = os.listdir(img_path)
# print(filelist)

# softmax layer for defect interpretation
softmax = torch.nn.Softmax(dim=0)

# this section loads in the weights of an already trained model
model = torchvision.models.segmentation.__dict__[pre_model](num_classes=classes,
                                                            aux_loss=aux_loss,
                                                            pretrained=False)
model = model.cuda()
checkpoint = torch.load(model_path+model_name, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

# transforms to put images through the model
trans = t.Compose([t.ToTensor(), t.Normalize(mean=.5, std=.2)])

for i in range(len(filelist)):
    # opens up and preps image, runs through model
    img = Image.open(img_path+filelist[i]).convert('RGB')
    img = trans(img).unsqueeze(0).cuda()
    output = model(img)['out']

    # threshold to determine defect vs. non-defect instead of softmax
    soft = softmax(output[0].cpu())
    nodef = soft[0]
    prob = copy.deepcopy(nodef.detach().numpy())
    nodef[nodef < threshold] = -1
    nodef[nodef >= threshold] = 0
    nodef = nodef.type(torch.int)
    def_idx = soft[1:].argmax(0).type(torch.int)
    def_idx = def_idx + 1
    nodef[nodef == -1] = def_idx[nodef == -1]

    output_pix = np.count_nonzero(nodef)
    output_defect_percent = output_pix / float(np.size(nodef.cpu().numpy().flatten()))

    # plots the original image next to prediction (with defect percentage)
    orig_img = (img * .2) + .5
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img[0][0].cpu().numpy(), cmap='inferno', vmin=0, vmax=1)
    plt.title('input')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(nodef.squeeze(), cmap=matplotlib.cm.get_cmap("magma", 4), vmin=0, vmax=2)
    plt.title('prediction')
    plt.tick_params(axis='both', labelsize=0, length=0)
    plt.xlabel("Defect Percentage: " + str(round(output_defect_percent, 5)))
    # plt.savefig(save_path + str(i) + '.png') # comment back in to save figures
    plt.show()
