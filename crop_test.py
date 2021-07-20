import cell_cropping
import cv2
import os


img_path = 'modules/'
modules = os.listdir(img_path)

for i in range(len(modules)):
    if os.path.isdir(img_path+modules[i]):
        continue

    try:
        cell_cropping.CellCropComplete(img_path + modules[i], i=i, NumCells_y=6, NumCells_x=12, corners_get='auto')
    except cv2.error:
        print('This module needs manual corner finding. Click each of the four corners, then press \'c\'. '
              'In case of mistake, please press \'r\' to reset corners.')
        cell_cropping.CellCropComplete(img_path + modules[i], i=i, NumCells_y=6, NumCells_x=12, corners_get='manual')
