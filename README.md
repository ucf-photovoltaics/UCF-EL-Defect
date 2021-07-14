This is a script meant for easy use of our defect segmentation model. Our models are trained and evaluated using the PyTorch python package. 

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

TO USE:
1. Place model path in the 'models/' folder and the images to test in the 'Test_Images/' directory.
2. Rename all the paths and folder names appropriately at the top of ELDefectSegmentation.py (if necessary).
3. Set model parameters according to the model used.
4. Run ELDefectSegmentation.py. To save the output, uncomment line #135.
 

Note:
- To save defect percentages and display them in plots, set defect_per (line #21) to True
    - Defects will be saved in defect\_dir (line #20) folder - currently 'defect\_percentages/'
- Adjust threshold (line #26) to test out custom defect thresholding (if the model gives the chance of a pixel to be defect-free  > threshold, set the pixel to defect-free. Otherwise, choose the highest possibility defect)
- Currently, module images will be automatically segmented into cells assuming 72 cells (6 x 12 cells)
- If cropped cells aren't satisfactory, try switching corners_get='auto' to 'manual' in line #78
    - This will display the module image, allowing you to manually click the four corners. Use 'r' to reset in case of mistake. Once the corners are correct, hit 'c'.
- Alternatively, use the crop_test.py file to segment the module ahead of time, manually adding the cells to the 'Test\_Images' directory

