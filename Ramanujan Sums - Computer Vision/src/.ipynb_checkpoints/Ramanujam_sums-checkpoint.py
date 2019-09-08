# To import the libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, sys
from math import gcd
import scipy.signal as signal


# ************funtion to find Ramanujan_SUMS *********** 

def Ramanujan_SUMS(q_val):
                    # --- To initiallize the y vector -------------------------------
                    #definition the array variable to save the Ramanujan sum
    y_val = []

    # to initialize the Ramanujan sum into zero at beginning
    for n in range(q_val):

        #add the zeros with float type in vector
        y_val.append(0.0)

    #to convert list type variable into array type to calculate the Ramanujan sum very easily.
    y_val = np.array(y_val)

    # repeat the loop to calculate the Ramanuian sums --------------------------
    for n in range(q_val):
        # to initialize the y value into 0
        y_val[n] = 0

        # to initialize m_value into 0 to calculate the sum
        mat_val = 0

        # calculate the sum while the loop repeats
        for x in range(1, q_val):

            # decide the condition whether x and q_val is greatest common divider or not
            if gcd(x, q_val) == 1:
                # if x and q_val is gcd, calculate the sum

                #to get the PI
                pi = 4 * np.arctan(1.0)

                # calculate the sum
                mat_val = mat_val + np.cos(2 * pi * x * n / q_val)

        # To save the calculated sum into vector
        y_val[n] = mat_val

    # return Ramanujan sum array
    return y_val
    

# ////////// main function to solve the problem /////////////////////////////////////////////
def main():

    # define the q_val_val value for ramanujan sum
    q_val = 3
    # define the variable to fix the image size
    IMAGE_WIDTH = 800

    #  ***********************************************************************************
    # set training flag
    train_flag = False
    
    # if training, set training variables and if not, load the trained classifiers
    if not (train_flag):
        # read the trained sample data
        sample = np.loadtxt('trainingData', np.float32)

        # read the trained response data
        response = np.loadtxt('label', np.float32)

        # convert the size of response data into flattern
        response = response.reshape((response.size, 1))

        # get the model of netwaork by using Kneast method
        model = cv2.ml.KNearest_create()

        # train the model by using training data
        model.train(sample, cv2.ml.ROW_SAMPLE, response)

    else:
        # define the array variable to save the response data
        response = []

        # define and initialize the array variable to save the sample data
        sample = np.empty((0, 10000))

        # define the array variable to get the key board value, while the training the model
        keys = [i for i in range(1, 180)]

    #****************************************************************************
    #To get the Ramanujan Sums
    rama_vec = Ramanujan_SUMS(q_val)

    # To construct Kernel matrix for the filter by using the Ramanujan Sums
    RS_keneral = np.zeros((q_val, q_val))
    for i in range(0, q_val):
        for j in range(len(RS_keneral)):
            RS_keneral[i, np.mod(i + j, q_val)] = rama_vec[j]

    # ****************************************************************************
    # set the file path
    images_dir = 'data/test/'
    filepath = [images_dir + f for f in os.listdir(images_dir)]


    for path in filepath:
        try:
            # read the image according with the path
            inputImg = cv2.imread(path,0)

            # to convert the read image into fixed size
            inputImg = cv2.resize(inputImg,dsize=(IMAGE_WIDTH,IMAGE_WIDTH))

            # to apply the Gaussian smoothing filter in read image
            inputImg = cv2.GaussianBlur(inputImg,(5,5),0)

            #--- to adjust the image bright and contrast ------------------------------------
            hist_img = cv2.equalizeHist(inputImg)

            # to convert gray image to bw image
            ret, bw_img = cv2.threshold(hist_img, 90, 255, cv2.THRESH_BINARY)

            # to apply the Ramanujan sum to detect the edge of image in bw image -------------------------
            BW_img = signal.convolve2d(bw_img, RS_keneral, 'same')

            # To get absolute value image
            BW_img = np.abs(BW_img)

            # to convert gray image to bw image
            ret, BW_img = cv2.threshold(BW_img, 1, 255, cv2.THRESH_BINARY)

            # ----------------------------------------------------------
            roi1 = np.copy(BW_img)

            # ------ convert the size for training -------------
            roi = cv2.resize(roi1, (100, 100))

            # ------ Convert the 2D matrix into flattern vector -------------
            roismall = roi.reshape((1, 10000))
            roismall = np.float32(np.array(roismall).astype(np.float32))

            # ------if training, get the label and sample ------------------------
            if train_flag:
                cv2.imshow('norm', roi1)
                key = cv2.waitKey(0)

                if key == 27:  # (escape to quit)

                    sys.exit()
                elif key in keys:

                    response.append(int(key))
                    sample = np.append(sample, roismall, 0)
            else:
                ret, results, neighbours, dist = model.findNearest(roismall, len(response))
                return_val = np.int16(chr(int(neighbours[0][0])))

                plt.subplot(1, 2, 1), plt.imshow(inputImg, 'gray'), plt.title('Input image')

                if(return_val==1):

                    plt.subplot(1, 2, 2), plt.imshow(BW_img, 'gray'), plt.title('Result= No TuberCulosis')
                    print(path+" : No TB ")
                else:

                    plt.subplot(1, 2, 2), plt.imshow(BW_img, 'gray'), plt.title('Result= TuberCulosis')
                    print(path + " : TB ")

                plt.show()

        except ValueError:
            continue

    # ------------if training, Save training data ---------------------------
    if train_flag:

        # To conver the list type data into array data
        response = np.array(response)
        response = response.reshape((response.size, 1))

        # save the sample data to use in model training
        np.savetxt('trainingData', sample, fmt='%0.1f')

        # save the response data to use in model training
        np.savetxt('label', response, fmt='%0.1f')



# //////// initializing funtoin /////////////////////
if __name__ == "__main__":
    # call the main funtion
    main()