# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 12:09:58 2018

@authors: Eric, Dylan
"""
import skimage.filters as filters
import numpy as np
import cv2
from pyhull import qconvex
from operator import attrgetter
from scipy import stats as st


###########################################################################

def resize_Image(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def define_Intersection(L1, L2):
    # determines the intersection of two lines
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def define_Line(p1, p2):
    # defines the parameters of a line from two points
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def detect_ModuleEdges(image, blurring_steps=12, display_houghlines=False, display_number=20):
    # Convert image to grayscale ***Not needed for InGaAs Camera
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image

    # Process image through a sequence of blurring
    for n in range(1, blurring_steps):
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply canny edge detection on blurred image
    edged = cv2.Canny(gray, 1, 75)

    # finds the first and last bright pixel in the vertical and horizontal directions
    # deletes anything in between these two. This allows for a broader range of blurring steps to work properly
    new_edged = np.zeros([len(edged[:, 0]), len(edged[0, :])])

    for i1 in range(len(edged[0, :])):

        idx_first = np.argmax(edged[:, i1])
        idx_last = len(edged[:, i1]) - np.argmax(edged[:, i1][::-1]) - 1
        if idx_first != 0:
            new_edged[idx_first, i1] = 255
        if idx_last != len(edged[:, i1]) - 1:
            new_edged[idx_last, i1] = 255

    for i1 in range(len(edged[:, 0])):

        idx_first = np.argmax(edged[i1, :])
        idx_last = len(edged[i1, :]) - np.argmax(edged[i1, :][::-1]) - 1
        if idx_first != 0:
            new_edged[i1, idx_first] = 255
        if idx_last != len(edged[i1, :]) - 1:
            new_edged[i1, idx_last] = 255

    edged = new_edged.astype(np.uint8)
    # Initialize arrays for line detection
    strong_lines = np.zeros([4, 1, 2])
    strong_lines_2D = np.zeros([4, 1, 2])

    # Use HoughLines function to detect lines in the edge detection image
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLines(edged, 1, np.pi / 180, 10, minLineLength, maxLineGap)

    # Try to verify that we did not detect two lines that essentially overlap (Not always perfect!)
    n4 = 0
    n5 = 0
    for n3 in range(0, len(lines)):

        for rho, theta in lines[n3]:

            # Check is theta is near normal as we expect a close to square EL Image
            if rho < 0:
                rho *= -1
                theta -= np.pi
            parallel_theta = np.isclose(theta, 0, atol=np.pi / 36)
            perpendicular_theta = np.isclose(theta, np.pi / 2, atol=np.pi / 36)
            # parallel_theta = True
            # perpendicular_theta = True
            if parallel_theta or perpendicular_theta:
                # if n3<100:
                # print(rho,theta)
                if n4 == 0:
                    strong_lines[n4] = lines[n3]
                    strong_lines_2D[n5] = lines[n3]
                    n4 = n4 + 1
                    n5 = n5 + 1

                else:
                    # is this line close close in terms of rho
                    closeness_rho = np.isclose(rho, strong_lines[0:n4, 0, 0], atol=100)

                    # is this line close in terms of theta
                    closeness_theta = np.isclose(theta, strong_lines[0:n4, 0, 1], atol=np.pi / 36)

                    # compare both theta and rho
                    closeness = np.all([closeness_rho, closeness_theta], axis=0)
                    directionality_theta = np.isclose(theta, strong_lines_2D[0:2, 0, 1], atol=np.pi / 9)
                    # if n3<100:
                    # print(n4,n5, closeness, directionality_theta)
                    if not any(closeness) and n4 < 4:
                        strong_lines[n4] = lines[n3]
                        n4 = n4 + 1
                    if not any(closeness) and n5 < 2:
                        strong_lines_2D[n5] = lines[n3]
                        n5 = n5 + 1

                    elif not any(directionality_theta) and n5 == 2:
                        strong_lines_2D[n5] = lines[n3]
                        n5 = n5 + 1

                    elif not any(directionality_theta) and n5 == 3:
                        closeness_rho_2D = np.isclose(rho, strong_lines_2D[2, 0, 0], atol=10)
                        if not closeness_rho_2D:
                            strong_lines_2D[n5] = lines[n3]
                            n5 = n5 + 1

    # print(strong_lines_2D)

    if display_houghlines:

        image_houghlines = gray
        for i in range(display_number):
            for rho, theta in lines[i]:
                # print(rho,theta)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 10000 * (-b))
                y1 = int(y0 + 10000 * (a))
                x2 = int(x0 - 10000 * (-b))
                y2 = int(y0 - 10000 * (a))

                cv2.line(image_houghlines, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("HoughLines", image_houghlines)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return strong_lines, gray, edged


def click_edges(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, event_num
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt[event_num, :] = [x, y]
        event_num += 1


def extract_CellGridAndModuleCorners(CVImage, DisplayImage, NumCells_x, NumCells_y, blurring_steps=None,
                                     resize_height=500, method="houghlines", perspective_correct=False):
    # Resize image for more efficient processing
    ratio = CVImage.shape[0] / resize_height
    image_Original = DisplayImage.copy()
    image_HoughLines = resize_Image(CVImage, height=resize_height)
    image_Outline = resize_Image(DisplayImage, height=resize_height)
    image_Reduced = resize_Image(CVImage, height=resize_height)

    # image_HoughLines = delete_frame(image_HoughLines,pixels=2)
    # image_Outline = delete_frame(image_Outline,pixels=2)
    # image_Reduced = delete_frame(image_Reduced,pixels=2)

    '''
    cv2.imshow("Edged",image_Reduced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # Apply edge detection function ***Beware this does not always work!***
    if blurring_steps == None:
        lines, image_Blurred, image_Threshold = detect_ModuleEdges(image_HoughLines)
    else:
        lines, image_Blurred, image_Threshold = detect_ModuleEdges(image_HoughLines, blurring_steps=blurring_steps)

    if method == "houghlines":
        # Convert edges into module corners
        lines_xy = np.zeros([4, 2, 2], dtype=np.float32)

        for i in range(0, 4):
            for rho, theta in lines[i]:
                # print(rho,theta)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 10000 * (-b))
                y1 = int(y0 + 10000 * (a))
                x2 = int(x0 - 10000 * (-b))
                y2 = int(y0 - 10000 * (a))

                cv2.line(image_HoughLines, (x1, y1), (x2, y2), (0, 0, 255), 2)

                lines_xy[i, 0, 0] = x1
                lines_xy[i, 0, 1] = y1
                lines_xy[i, 1, 0] = x2
                lines_xy[i, 1, 1] = y2

                # cv2.imshow("HoughLines", image_HoughLines)
                # cv2.imshow("Burred", image_Blurred)
                # cv2.imshow("Edged",image_Threshold)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

        corners = np.zeros([4, 2], dtype=np.float32)
        corners[0, :] = define_Intersection(define_Line(lines_xy[0, 0, :], lines_xy[0, 1, :]),
                                            define_Line(lines_xy[2, 0, :], lines_xy[2, 1, :]))
        corners[1, :] = define_Intersection(define_Line(lines_xy[0, 0, :], lines_xy[0, 1, :]),
                                            define_Line(lines_xy[3, 0, :], lines_xy[3, 1, :]))
        corners[2, :] = define_Intersection(define_Line(lines_xy[1, 0, :], lines_xy[1, 1, :]),
                                            define_Line(lines_xy[2, 0, :], lines_xy[2, 1, :]))
        corners[3, :] = define_Intersection(define_Line(lines_xy[1, 0, :], lines_xy[1, 1, :]),
                                            define_Line(lines_xy[3, 0, :], lines_xy[3, 1, :]))

        corners = order_ModuleCorners(corners)
        # print(corners)

    elif method == "manual":

        clone = image_Reduced.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_edges)
        global event_num
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", image_Reduced)
            key = cv2.waitKey(1) & 0xFF

            cv2.circle(image_Reduced, (refPt[event_num - 1, 0], refPt[event_num - 1, 1]), 2, (0, 255, 0), thickness=2)
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                image_Reduced = clone.copy()
                event_num = 0
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break

        corners = order_ModuleCorners(refPt)

    # Draw module edge lines on display image
    cv2.line(image_Outline, (corners[0, 0], corners[0, 1]), (corners[1, 0], corners[1, 1]), (0, 0, 255), 2)
    cv2.line(image_Outline, (corners[1, 0], corners[1, 1]), (corners[2, 0], corners[2, 1]), (0, 0, 255), 2)
    cv2.line(image_Outline, (corners[2, 0], corners[2, 1]), (corners[3, 0], corners[3, 1]), (0, 0, 255), 2)
    cv2.line(image_Outline, (corners[3, 0], corners[3, 1]), (corners[0, 0], corners[0, 1]), (0, 0, 255), 2)

    # cv2.imshow("Outline", image_Outline)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if perspective_correct:
        # TODO:If module is not aligned with image edges apply a tranformation ****This step has been excluded****
        # image_Transformed = transform_Image(image_Original, corners*ratio)
        # now that we have our rectangle of points, let's compute
        # the width of our new image
        (tl, tr, br, bl) = corners * ratio
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        # ...and now for the height of our new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        # take the maximum of the width and height values to reach
        # our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        # construct our destination points which will be used to
        # map the screen to a top-down, "birds eye" view
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen
        M = cv2.getPerspectiveTransform(corners * ratio, dst)
        image_warped = cv2.warpPerspective(image_Original, M, (maxWidth, maxHeight))

        cv2.imshow("PerspectiveCorrection", image_warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        corners_warped = np.zeros([4, 2], dtype=np.float32)
        corners_warped[0, :] = [maxWidth, 0]
        corners_warped[1, :] = [0, maxHeight]
        corners_warped[2, :] = [0, 0]
        corners_warped[3, :] = [maxWidth, maxHeight]

        corners_warped = order_ModuleCorners(corners_warped)

        gridPoints = extract_GridPoints(corners_warped, (NumCells_x, NumCells_y))
        image_warped_outline = image_warped.copy()
        # Draw grid on display image
        for x0, y0 in gridPoints:
            cv2.circle(image_warped_outline, (int(x0), int(y0)), 2, (0, 255, 0), thickness=2)
        image_Outline = image_warped_outline
        image_Original = image_warped

        cv2.imshow("Outline", image_Outline)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Create grid from module corners and number of cells in X and Y direction
    else:
        gridPoints = extract_GridPoints(corners, (NumCells_x, NumCells_y))
        # print(gridPoints)
        # Draw grid on display image
        for x0, y0 in gridPoints:
            cv2.circle(image_Outline, (int(x0), int(y0)), 2, (0, 255, 0), thickness=2)

        # Translate corners and grid locations to original image size
        gridPoints = gridPoints * ratio
        corners = corners * ratio
    # print(corners)
    return image_Outline, gridPoints, corners, image_Original


refPt = np.zeros([4, 2], dtype=np.float32)
event_num = 0


def order_ModuleCorners(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def extract_GridPoints(edges, nCells, dtype=float):
    # creates a regular 2d grid from given edge points (4*(x0,y0))
    # and number of cells in x and y
    # returns horizontal and vertical lines as (x0,y0,x1,y1)

    e = order_ModuleCorners(edges)
    sx, sy = nCells[0] + 1, nCells[1] + 1
    # horizontal lines
    x0 = np.linspace(e[0, 0], e[3, 0], sy, dtype=dtype)
    x1 = np.linspace(e[1, 0], e[2, 0], sy, dtype=dtype)

    y0 = np.linspace(e[0, 1], e[3, 1], sy, dtype=dtype)
    y1 = np.linspace(e[1, 1], e[2, 1], sy, dtype=dtype)
    # points:
    p = np.empty(shape=(sx * sy, 2))
    n0 = 0
    n1 = sx
    for x0i, x1i, y0i, y1i in zip(x0, x1, y0, y1):
        p[n0:n1, 0] = np.linspace(x0i, x1i, sx)
        p[n0:n1, 1] = np.linspace(y0i, y1i, sx)
        n0 = n1
        n1 += sx
    return p


def CellCropping(img, NumCells_x=10, NumCells_y=6, border_width=200):
    """Crops cells from module image

    Args:
        img (numpy.ndarray): An image array
        border_width (int): Integer for black border around image. Useful when
        modules are close to sensor edges
        NumCells_x (int): Number of cells in the x-direction (number of columns)
        NumCells_y (int): Number of cells in the y-direction (number of rows)
    Returns:
        transformedImg numpy.ndarray
        cells  list of numpy.ndarrays
        boundaries list of boundaries used for cropping; top, bottom, left, right
    """

    def Mask(img):
        """Creates a mask of the cell area.

        Thresholds the image to create a binary mask.

        Args:
            img (numpy.ndarray): An image array
        Returns:
            numpy.ndarray
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img_gray = rgb2gray(img)
        img_gray = (img_gray) ** (1 / .35)
        # img_gray = (img_gray)**(1.6)
        grayThreshold = filters.threshold_otsu(img_gray)
        mask = img_gray > grayThreshold
        mask = mask.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        return mask

    def CellExtract(img, numCols, numRows):
        """Performs extraction of individual cells.

        Args:
            img (numpy.ndarray): An image array
            numCols (int): number of cells across in module image
            numRows (int): number of cells down in module image

        Returns:
            list of numpy.ndarrays
        """
        mask = Mask(img)
        cols = np.any(mask, axis=0)
        rows = np.any(mask, axis=1)
        xmin, xmax = min(np.where(cols)[0]), max(np.where(cols)[0])
        ymin, ymax = min(np.where(rows)[0]), max(np.where(rows)[0])
        # Detecting rows and columns of the module
        colsums = np.sum(mask, axis=0)
        cellwidth = (xmax - xmin) / numCols
        midpts = [int(i * cellwidth + xmin) for i in range(1, numCols)]
        xcuts = [np.median(np.where(colsums[int(midpt - cellwidth / 5):int(midpt + cellwidth / 5)] == colsums[
                                                                                                      int(midpt - cellwidth / 5):int(
                                                                                                          midpt + cellwidth / 5)].min())[
                               0]) + int(midpt - cellwidth / 5) for midpt in midpts]
        row_ch_pts = [0] + list(map(int, xcuts)) + [len(cols)]
        rowsums = np.sum(mask, axis=1)
        cellheight = (ymax - ymin) / numRows
        hmidpts = [int(i * cellheight + ymin) for i in range(1, numRows)]
        ycuts = [np.median(np.where(rowsums[int(midpt - cellheight / 5):int(midpt + cellheight / 5)] == rowsums[
                                                                                                        int(midpt - cellheight / 5):int(
                                                                                                            midpt + cellheight / 5)].min())[
                               0]) + int(midpt - cellheight / 5) for midpt in hmidpts]
        col_ch_pts = [0] + list(map(int, ycuts)) + [len(rows)]
        # Separating the module into cell arrays
        cellarrays = []
        for i in range(numRows):
            for j in range(numCols):
                result = img[col_ch_pts[i]:col_ch_pts[i + 1], row_ch_pts[j]:row_ch_pts[j + 1]]
                cellarrays.append(result)

        return cellarrays

    # Adding border around image for images in which the module is close to
    # the sensor edge, i.e., the edge of the image. This prevents cropping errors.

    # img = cv2.copyMakeBorder(img,border_width,border_width,border_width,border_width,cv2.BORDER_CONSTANT)

    # Getting the approximate size of the image
    mask = Mask(img)
    hullDims = mask.shape
    if mask[0, :].any() == True:
        mask = np.concatenate([np.zeros((hullDims[0], 10), dtype='uint8'), mask], axis=1)
        img = np.concatenate([np.zeros((hullDims[0], 10, 3), dtype='uint8'), img], axis=1)
        hullDims = mask.shape
    if mask[-1, :].any() == True:
        mask = np.concatenate([mask, np.zeros((hullDims[0], 10), dtype='uint8')], axis=1)
        img = np.concatenate([img, np.zeros((hullDims[0], 10, 3), dtype='uint8')], axis=1)
        hullDims = mask.shape
    if mask[:, 0].any() == True:
        mask = np.concatenate([np.zeros((10, hullDims[1]), dtype='uint8'), mask], axis=0)
        img = np.concatenate([np.zeros((10, hullDims[1], 3), dtype='uint8'), img], axis=0)
        hullDims = mask.shape
    if mask[:, -1].any() == True:
        mask = np.concatenate([mask, np.zeros((10, hullDims[1]), dtype='uint8')], axis=0)
        img = np.concatenate([img, np.zeros((10, hullDims[1], 3), dtype='uint8')], axis=0)
        hullDims = mask.shape
    for i in range(hullDims[0]):
        row = mask[i, :]
        if row.any() == True:
            startRow = i
            break
    for i in reversed(range(hullDims[0])):
        row = mask[i, :]
        if row.any() == True:
            endRow = i
            break
    for i in range(hullDims[1]):
        col = mask[:, i]
        if col.any() == True:
            startCol = i
            break
    for i in reversed(range(hullDims[1])):
        col = mask[:, i]
        if i:
            endCol = i
            break
    midRows = int((endRow + startRow) / 2)
    midCols = int((endCol + startCol) / 2)
    maskpoints = np.column_stack(np.nonzero(mask))

    # Performing the convex hull and merging facets with less than 0.5% change in slope
    vertices = qconvex('A0.99995 PM50 i', maskpoints)
    vertarray = np.zeros((len(vertices), 2), dtype=int)
    for i in range(len(vertices) - 1):
        vertarray[i + 1] = list(map(int, vertices[i + 1].split(' ')))[:2]
    vert = maskpoints[vertarray]
    vert = vert[:, :, :2]

    # Classifying facets into 4 sides
    top = []
    bottom = []
    left = []
    right = []
    for i in range(len(vert)):
        y, x = vert[i].T
        if (abs(y[0] - y[1]) < abs(x[0] - x[1])):  # horizontal
            if y[0] < midRows:
                top.append(vert[i])
            elif y[0] > midRows:
                bottom.append(vert[i])
        elif (abs(y[0] - y[1]) > abs(x[0] - x[1])):  # vertical
            if x[0] < midCols:
                left.append(vert[i])
            elif x[0] > midCols:
                right.append(vert[i])

    # Getting the longest facet for each side
    points = []
    for side in (top, left, bottom, right):
        lengths = []
        for i in range(len(side)):
            y, x = side[i].T
            lengths.append(np.sqrt(abs(y[0] - y[1]) ** 2 + abs(x[0] - x[1]) ** 2))
        points.append(side[lengths.index(max(lengths))])
    # Finding intercepts of the 4 facets
    intercepts = []
    for j in ((0, 1), (1, 2), (2, 3), (3, 0)):
        y1, x1 = points[j[0]].T.astype(float)
        y2, x2 = points[j[1]].T.astype(float)
        slopeh, inth = attrgetter('slope', 'intercept')(st.linregress(x1, y1))
        if x2[0] == x2[1]:
            x = x2[0]
        else:
            slopev, intv = attrgetter('slope', 'intercept')(st.linregress(x2, y2))
            x = (intv - inth) / (slopeh - slopev)
        y = slopeh * x + inth
        intercepts.append([x, y])
    intercepts = np.asarray(intercepts).astype(int)
    x, y = intercepts.T

    pts = np.float32([[x[0], y[0]], [x[3], y[3]], [x[2], y[2]], [x[1], y[1]]])
    xcrop = list(map(int, pts[:, 0]))
    ycrop = list(map(int, pts[:, 1]))
    xcrop.sort()
    ycrop.sort()

    if ycrop[0] < 0:
        ycrop[0] = int(-1 * ycrop[0])
    if xcrop[0] < 0:
        xcrop[0] = int(-1 * xcrop[0])

        # Cropping edges are taken as the average between the two corners of each side
    topCrop = int(np.mean([ycrop[0], ycrop[1]]))
    bottomCrop = int(np.mean([ycrop[2], ycrop[3]]))
    leftCrop = int(np.mean([xcrop[0], xcrop[1]]))
    rightCrop = int(np.mean([xcrop[2], xcrop[3]]))
    boundaries = [topCrop, bottomCrop, leftCrop, rightCrop]
    # boundaries = [b - border_width for b in boundaries] # NECESSARY for proper cropping using this output

    corners = np.zeros([4, 2])
    corners[0, :] = [boundaries[2], boundaries[0]]
    corners[1, :] = [boundaries[3], boundaries[0]]
    corners[2, :] = [boundaries[3], boundaries[1]]
    corners[3, :] = [boundaries[2], boundaries[1]]
    # Cropped image to be used as input for cell cropping function
    transformedImg = img[topCrop:bottomCrop, leftCrop:rightCrop]

    # cells = CellExtract(transformedImg, NumCells_x,NumCells_y)
    gridPoints = extract_GridPoints(corners, (NumCells_x, NumCells_y))

    return transformedImg, corners, gridPoints


import os


# Here is what you're looking for
# This function crops cells from module image
# Auto
def CellCropComplete(image_file, i='', NumCells_x=12, NumCells_y=6, corners_get='auto'):
    filepath_cell_images = os.path.dirname(image_file) + '/Cell_Images' + str(i) + '/'
    filepath_cell_images_enhanced = os.path.dirname(image_file) + '/Cell_Images_Enhanced/'

    if not os.path.isdir(filepath_cell_images):
        os.mkdir(filepath_cell_images)
    # if not os.path.isdir(filepath_cell_images_enhanced):
    #     os.mkdir(filepath_cell_images_enhanced)

    DisplayImage = cv2.imread(image_file)
    CVImage = DisplayImage.copy()
    NumCells = NumCells_x * NumCells_y

    if corners_get == 'manual':
        # gridPoints = extract_GridPoints(corners, (NumCells_x,NumCells_y))
        image_Outline, gridPoints, corners, image_Original = extract_CellGridAndModuleCorners(
            CVImage, DisplayImage, NumCells_x, NumCells_y, resize_height=500, method="manual",
            perspective_correct=False)
    elif corners_get == 'auto':
        # croppedimage, corners, gridPoints = CellCropping(DisplayImage, NumCells_x=NumCells_x, NumCells_y=NumCells_y)
        # Use what is commented out below if for some reason, auto is not working
        image_Outline, gridPoints, corners, image_Original = extract_CellGridAndModuleCorners(CVImage, DisplayImage,
                                                                                              NumCells_x, NumCells_y,
                                                                                              resize_height=500,
                                                                                              perspective_correct=False)

    # Writing image files for cropped cells, cropped cells enhanced
    DisplayImage_L = np.zeros(NumCells, dtype=int)
    DisplayImage_R = np.zeros(NumCells, dtype=int)
    DisplayImage_T = np.zeros(NumCells, dtype=int)
    DisplayImage_B = np.zeros(NumCells, dtype=int)

    for i2 in range(0, NumCells_y):
        for i3 in range(0, NumCells_x):
            i4 = NumCells_x * i2 + i3 + i2
            i5 = NumCells_x * i2 + i3
            DisplayImage_L[i5] = int(gridPoints[i4, 0])
            DisplayImage_R[i5] = int(gridPoints[i4 + NumCells_x + 2, 0])
            DisplayImage_T[i5] = int(gridPoints[i4, 1])
            DisplayImage_B[i5] = int(gridPoints[i4 + NumCells_x + 2, 1])
            DisplayImage_CellCropped = DisplayImage[DisplayImage_T[i5]:DisplayImage_B[i5],
                                       DisplayImage_L[i5]:DisplayImage_R[i5]]
            cv2.imwrite(filepath_cell_images + '/cell_' + str(10 * i2 + i3 + 1) + '.jpg', DisplayImage_CellCropped)
            # cv2.imwrite(filepath_cell_images_enhanced + '/cell_' + str(10 * i2 + i3 + 1) + '.jpg',
            #             DisplayImage_CellCropped * 5)
    # print('\nBeta version - final version to be developed\n')
    return corners, gridPoints