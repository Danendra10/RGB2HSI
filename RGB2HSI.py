# Library Initialization
import math
import numpy as np
import cv2 as cv
#===============================================================================

# Function to convert RGB to HSI

#===============================================================================
#---------------------------First Algorithm-------------------------------------
#===============================================================================
def RGB2HSI(RGB):
    # Convert RGB to float
    RGB = RGB.astype(float)
    # Get the dimensions of the image
    rows, cols, channels = RGB.shape
    # Initialize the HSI image
    HSI = np.zeros((rows, cols, channels))
    # Convert RGB to HSI
    for i in range(rows):
        for j in range(cols):
            # Get the RGB values
            R = RGB[i, j, 0]
            G = RGB[i, j, 1]
            B = RGB[i, j, 2]
            # Calculate the HSI values
            I = (R + G + B) / 3
            S = 1 - (3 / (R + G + B)) * min(R, G, B)
            H = math.acos((0.5 * ((R - G) + (R - B))) / (math.sqrt((R - G) ** 2 + (R - B) * (G - B))))
            if B > G:
                H = 2 * math.pi - H
            # Store the HSI values
            HSI[i, j, 0] = H
            HSI[i, j, 1] = S
            HSI[i, j, 2] = I
    # Return the HSI image
    return HSI

#===============================================================================
#---------------------------Second Algorithm------------------------------------
#===============================================================================
def RGB2HSI2P0(img):
    with np.errstate(divide='ignore', invalid='ignore'):

        #Load image with 32 bit floats as variable type
        bgr = np.float32(img)/255

        #Separate color channels
        blue = bgr[:,:,0]
        green = bgr[:,:,1]
        red = bgr[:,:,2]

        #Calculate Intensity
        def calc_intensity(red, blue, green):
            return np.divide(blue + green + red, 3)

        #Calculate Saturation
        def calc_saturation(red, blue, green):
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)

            return saturation

        #Calculate Hue
        def calc_hue(red, blue, green):
            hue = np.copy(red)

            for i in range(0, blue.shape[0]):
                for j in range(0, blue.shape[1]):
                    hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                                math.sqrt((red[i][j] - green[i][j])**2 +
                                        ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                    hue[i][j] = math.acos(hue[i][j])

                    if blue[i][j] <= green[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]

            return hue

        #Merge channels into picture and return image
        hsi = cv.merge((calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_intensity(red, blue, green)))
        return hsi