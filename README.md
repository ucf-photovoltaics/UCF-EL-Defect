# Semantic Segmentation of Photovoltaic Cells
This is a script meant for easy use of our defect segmentation model, seen in our paper [Automated Defect Detection and Localization in Photovoltaic Cells using Semantic Segmentation of Electroluminescence Images](https://ieeexplore.ieee.org/document/9650542). Our models are trained and evaluated using the PyTorch python package. 

The images of the dataset are all individual cell images found in the 'Test_Images/' folder, and the .csv containing annotations are found in 'annotations.csv'. These are exported straight from the annotation software, then combined from different annotation sessions. The training scripts and images are found in the 'training/' folder. The 'test_images.csv' and 'train_images.csv' contain the filenames of our train-test split evaluated in the paper.

The data can be used separately from the code, but evaluation code is provided. Brief instructions for use are found below. It can be used on our images, or any cell images to test for accuracy with our specific method. NOTE: The given model was trained for specific defects (cracks, contact defects, interconnect issues, and corrosion) on the module types included in our dataset. 

# TO USE:
1. Place model path in the 'models/' folder and the images to test in the 'Test_Images/' directory.
2. Rename all the paths and folder names appropriately at the top of ELDefectSegmentation.py (if necessary).
3. Set model parameters according to the model used.
4. Run ELDefectSegmentation.py. To save the output, uncomment line #143.

This code requires the following python packages:
- cv2
- json
- matplotlib
- numpy
- operator
- os
- scipy
- skimage
- torch
- torchvision
- PIL

Note:
- To save defect percentages and display them in plots, set defect_per (line #21) to True
    - Defects will be saved in defect\_dir (line #20) folder - currently 'defect\_percentages/'
- Adjust threshold (line #26) to test out custom defect thresholding (if the model gives the chance of a pixel to be defect-free  > threshold, set the pixel to defect-free. Otherwise, choose the highest possibility defect)
- Currently, module images will be automatically segmented into cells assuming 72 cells (6 x 12 cells)
- If cropped cells aren't satisfactory, try switching corners_get='auto' to 'manual' in line #80
    - This will display the module image, allowing you to manually click the four corners. Use 'r' to reset in case of mistake. Once the corners are correct, hit 'c'.
- Alternatively, use the crop_test.py file to segment the module ahead of time, manually adding the cells to the 'Test\_Images' directory


## Bibtex:
```
@article{fioresi2022automated,
  author={Fioresi, Joseph and Colvin, Dylan J. and Frota, Rafaela and Gupta, Rohit and Li, Mengjie and Seigneur, Hubert P. and Vyas, Shruti and Oliveira, Sofia and Shah, Mubarak and Davis, Kristopher O.},
  journal={IEEE Journal of Photovoltaics}, 
  title={Automated Defect Detection and Localization in Photovoltaic Cells Using Semantic Segmentation of Electroluminescence Images}, 
  year={2022},
  volume={12},
  number={1},
  pages={53-61},
  doi={10.1109/JPHOTOV.2021.3131059}}
```
