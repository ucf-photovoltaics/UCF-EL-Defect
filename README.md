This is a script meant for easy use of our defect segmentation model. Our models are trained and evaluated using the PyTorch python package. 

This code requires the following python packages:
- json
- matplotlib
- numpy
- os
- torch
- torchvision
- PIL

TO USE:
1. Place model path in the 'models/' folder and the images to test in the 'Test_Images/' directory.
2. Rename all the paths and folder names appropriately at the top of ELDefectSegmentation.py (if necessary).
3. Set model parameters according to the model used.
4. Run ELDefectSegmentation.py. To save the output, uncomment line #121.
 

Note:
- To save defect percentages and display them in plots, set defect_per (line #20) to True
    - Defects will be saved in defect\_dir (line #19) folder - currently 'defect\_percentages/'
- Adjust threshold (line #25) to test out custom defect thresholding (if the model gives the chance of a pixel to be defect-free  > threshold, set the pixel to defect-free. Otherwise, choose the highest possibility defect)

