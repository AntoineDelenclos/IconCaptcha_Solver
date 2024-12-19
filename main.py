# Importing different libraries
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

Path_folder = 'challenges/'

# This will allow us to change the path and to adapt more to where are the files and how much captcha there are
NUMBER_CAPTCHAS = len([f for f in os.listdir(Path_folder) if os.path.isfile(os.path.join(Path_folder, f))])

img_challenges = [cv2.imread(Path_folder + 'challenge_{}.png'.format(i), 0) for i in range(0, NUMBER_CAPTCHAS)]
binary_challenges = [[None, 0, 0, 0] for i in range(
    NUMBER_CAPTCHAS)]  # List of list that will contain the binary image, x_length, y_length and the number of icons per captcha
icons = []  # List of all the icons per captcha


# This function will detect all the icons inside a captcha
def detection_icons():
    for i in range(0, NUMBER_CAPTCHAS):
        _, binary_challenges[i][0] = cv2.threshold(img_challenges[i], 127, 255, cv2.THRESH_BINARY)  # Output is 2 dimension array but only the pixel's matrix is needed for us
        binary_challenges[i][1] = len(binary_challenges[i][0][0])  # x value
        binary_challenges[i][2] = len(binary_challenges[i][0])  # y value
        if binary_challenges[i][0][0][0] == 0:  # Background is black
            for k in range(binary_challenges[i][1]):
                for j in range(binary_challenges[i][2]):
                    binary_challenges[i][0][j][k] = np.abs(binary_challenges[i][0][j][k] - 255)  # This will switch all images in a white background, binarized, copy

        # We recognized a pattern, every captcha size depends on the number of icons inside them
        binary_challenges[i][3] = int(binary_challenges[i][1] * 0.05 - 16)  # Number of icon = Y * 0.05 - 16

        icon_size = binary_challenges[i][1] / binary_challenges[i][3]  # Size of each icon

        sublist_icons = []
        for z in range(binary_challenges[i][3]):  # For each captcha we will add all of its icons inside the corresponding sublist
            matrix = np.asmatrix(binary_challenges[i][0])
            start_col = int(icon_size * z)
            start_row = 0
            width = icon_size
            height = binary_challenges[i][2]
            icon_area = matrix[start_row:int(start_row + height), start_col:int(start_col + width)]  # We will just cut the global captcha in several similar sized areas
            sublist_icons.append(icon_area)

        icons.append(sublist_icons)

#This function will test all icons between each others to see if they are alone in the list or not
def transformation_icon(currentIconNumber, iconsList, nbTotalIcons, threshold):
    number_egal = 0  #is used to find if we found a second icon corresponding to the img_trait or not
    img_trait = iconsList[currentIconNumber] #the actual icon that is searched
    #we iterate on all icons of the Captcha and turning the icon in 360
    for i in range(4):
        for j in range(nbTotalIcons):
            if j == currentIconNumber:
                continue
            #we check is the image is egal to another
            result = egals_images(threshold, img_trait, iconsList[j])
            if result:
                #if it is egal to
                number_egal += 1
        img_trait = cv2.rotate(img_trait, cv2.ROTATE_90_CLOCKWISE)
    # We flip the image and then rotate it to compare (horizontal flip)
    img_trait = cv2.flip(img_trait, 1)
    for i in range(4):
        for j in range(nbTotalIcons):
            if j == currentIconNumber:
                continue
            result = egals_images(threshold, img_trait, iconsList[j])
            if result:
                number_egal += 1
        img_trait = cv2.rotate(img_trait, cv2.ROTATE_90_CLOCKWISE)
    # We flip again the image and then rotate it to compare (vertical flip)
    img_trait = cv2.flip(img_trait, 0)
    for i in range(4):
        for j in range(nbTotalIcons):
            if j == currentIconNumber:
                continue
            result = egals_images(threshold, img_trait, iconsList[j])
            if result:
                number_egal += 1
        img_trait = cv2.rotate(img_trait, cv2.ROTATE_90_CLOCKWISE)
    #We flip once again the image and rotate it to compare (both horizontal and vertical flip)
    img_trait = cv2.flip(img_trait, -1)
    for i in range(4):
        for j in range(nbTotalIcons):
            if j == currentIconNumber:
                continue
            result = egals_images(threshold, img_trait, iconsList[j])
            if result:
                number_egal += 1
        img_trait = cv2.rotate(img_trait, cv2.ROTATE_90_CLOCKWISE)

    return number_egal

#This function has the goal to centralize the searching of the true captcha and is using the transformation icon function
def search_captcha(list_icones):
    number_icones = 0
    captcha_found = False
    iteration = 0
    nmb_total_icones = len(list_icones)
    # Check if the captcha has found a correspondence, with a threshold of 5 for the image similarity (threshold = 5 means the comparaison between icons doesn't need to be exact)
    while not captcha_found and iteration < nmb_total_icones:
        #We check if we found another icon that is the same as the iteration_icons
        number_egal = transformation_icon(iteration, list_icones, nmb_total_icones, 5)
        #if it is egal to 0 it means that we didn't find any icon that correspond to the one we are iterating on
        if number_egal == 0:
            captcha_found = True
            number_icones = iteration + 1
        #else it means that we found a second icon that is the same, then we iterate on the next icon
        if captcha_found is False:
            iteration += 1
    #If we didn't find any correspondence, we retry but this time we are comparing the icons using a threshold of 1 (it means it is stricter on the comparaison)
    if number_icones == 0:
        iteration = 0
        while not captcha_found and iteration < nmb_total_icones:
            number_egal = transformation_icon(iteration, list_icones, nmb_total_icones, 1)
            if number_egal == 0:
                captcha_found = True
                number_icones = iteration + 1
            if captcha_found is False:
                iteration += 1
        if number_icones == 0:
            return 0
        else:
            print(f"Le captcha est l'icône numéro {number_icones}")
            return number_icones
    else:
        print(f"Le captcha est l'icône numéro {number_icones}")
        return number_icones

#This function is used to display the alone icon in the captcha
def reconstruction_CAPTCHA(number_captcha_break, captcha_image, number_icons):
    # We convert the image in color
    captcha_image = cv2.cvtColor(captcha_image, cv2.COLOR_GRAY2BGR)
    #We recover the size of the image
    height, width = captcha_image.shape[:2]
    # We compute the coordinate of the specified part (number_catcha_break)
    i = number_captcha_break - 1
    start_col = i * width // number_icons
    end_col = (i + 1) * width // number_icons
    top_left = (start_col, 0)
    bottom_right = (end_col - 1, height - 1)
    #then we draw a red rectangle on the alone icon, with a width of 3
    cv2.rectangle(captcha_image, top_left, bottom_right, (0, 0, 255), 3)

    # We convert the image in RGB
    captcha_image_rgb = cv2.cvtColor(captcha_image, cv2.COLOR_BGR2RGB)

    #Then we display the image
    plt.figure(figsize=(5, 5))
    plt.imshow(captcha_image_rgb)
    plt.xticks([]), plt.yticks([])
    plt.show()

#This function is comparing 2 images between them, with a threshold that allow the similarity to be exact or not (threshold bigger,  not precise)
def egals_images(threshold, image1, image2):
    #Check the size of the images
    if image1.size != image2.size:
        return "Error, the size of the images is not egal"
    #We do the difference between the 2 images
    difference = cv2.absdiff(image1, image2)
    difference = difference / 255  # 0 & 1 values inside the matrix
    pixelShiftNumber = np.sum(difference)
    #check if it corresponds or not with the treshold
    if pixelShiftNumber < threshold:
        return True
    else:
        return False

#The main function that execute all the code using the functions we wrote
def main():
    detection_icons()
    for i in range(NUMBER_CAPTCHAS):
        verif_captcha = search_captcha(icons[i])
        if verif_captcha == 0:
            print("Erreur en cherchant le Captcha")
        reconstruction_CAPTCHA(verif_captcha, img_challenges[i], len(icons[i]))

main()