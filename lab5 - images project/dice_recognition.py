import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def null_func(arg):
    pass


def draw_circles(image, circles):
    circles_np = np.round(circles[0, :]).astype('int')
    for (x, y, radius) in circles_np:
        cv2.circle(image, (x, y), radius, (255, 0, 0), 2)


def find_dots(dice, param1, param2, min_radius, max_radius):
    """Counts dots of a dice

    Args:
        dice (np.ndarray): a cropped image of a dice
        thresh (int, optional): the area below which a contour will be considered a dot. Defaults to 500.

    Returns:
        int: number of dots
    """
    dice_copy = dice.copy()

    dots = cv2.HoughCircles(dice_copy, cv2.HOUGH_GRADIENT, 1, 20, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)

    if dots is not None:
        return dots.reshape(-1, 3).astype(int)
    return None


def crop_dice(img, pos):
    """Crops a dice from an image in a given position

    Args:
        img (np.ndarray): an image from which the dices will be cropped
        pos (tuple): a tuple containing the position of the dice (x, y, w, h)

    Returns:
        np.ndarray: a cropped image of a dice
    """
    (x, y, w, h) = pos

    return img[y:y+h, x:x+w]


def plot_transformations(img_path='sample_images/dices.jpg'):
    """Plots all transformations, one by one, and saves the transformed images to a pdf.

    Args:
        img_path (str, optional): the path to an image. Defaults to 'dices.jpg'.
    """
    img_base = cv2.imread(img_path)
    img_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_base, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    _, img_thresh = cv2.threshold(img_blur, 195, 255, cv2.THRESH_BINARY)
    img_canny = cv2.Canny(img_thresh, 0, 0)

    images = [img_base, img_gray, img_blur, img_thresh, img_canny]
    transformations = ['Base', 'Grayscale', 'Gaussian Blur', 'Threshold', 'Canny']
    for i in range(5):
        plt.subplot(2, 3, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(transformations[i])
        plt.imshow(images[i], cmap='gray')
    plt.savefig('transformations.pdf')
    plt.show()



def main():
    """
    Creates windows with previews and trackbars to easily adjust parameters.
    """
    cv2.namedWindow('window', cv2.WINDOW_NORMAL)

    cv2.createTrackbar('grayscale', 'window', 0, 1, null_func)
    cv2.createTrackbar('gaussian_toggle', 'window', 0, 1, null_func)
    cv2.createTrackbar('kernel_gaussian', 'window', 1, 5, null_func)
    cv2.createTrackbar('sigma_gaussian', 'window', 0, 10, null_func)
    cv2.createTrackbar('threshold_toggle', 'window', 0, 1, null_func)
    cv2.createTrackbar('threshold', 'window', 0, 255, null_func)
    cv2.createTrackbar('canny_toggle', 'window', 0, 1, null_func)
    cv2.createTrackbar('contour_toggle', 'window', 0, 1, null_func)
    cv2.createTrackbar('contour_area', 'window', 500, 1000, null_func)
    cv2.createTrackbar('param1', 'window', 40, 100, null_func)
    cv2.createTrackbar('param2', 'window', 10, 100, null_func)
    cv2.createTrackbar('max_rad', 'window', 0, 100, null_func)

    base_img = cv2.imread('sample_images/zoomed/5.jpg')

    while True:
        img = base_img.copy()  # the copy of the base image that will be transformed
        img_2 = base_img.copy()  # the copy of the base image on which the contours etc will be drawn

        gray_bool       = cv2.getTrackbarPos('grayscale', 'window') == 1
        gaussian_bool   = cv2.getTrackbarPos('gaussian_toggle', 'window') == 1
        kernel_g        = cv2.getTrackbarPos('kernel_gaussian', 'window')
        sigma_g         = cv2.getTrackbarPos('sigma_gaussian', 'window')
        threshold_bool  = cv2.getTrackbarPos('threshold_toggle', 'window') == 1
        thresh          = cv2.getTrackbarPos('threshold', 'window')
        canny_bool      = cv2.getTrackbarPos('canny_toggle', 'window') == 1
        contour_bool    = cv2.getTrackbarPos('contour_toggle', 'window') == 1
        area            = cv2.getTrackbarPos('contour_area', 'window')  # the minimum area of a dice
        param_1         = cv2.getTrackbarPos('param1', 'window')
        param_2         = cv2.getTrackbarPos('param2', 'window')
        max_rad         = cv2.getTrackbarPos('max_rad', 'window')

        if gray_bool:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if gaussian_bool:
            if kernel_g % 2 == 1:
                img = cv2.GaussianBlur(img, (kernel_g, kernel_g), sigma_g)
        
        if threshold_bool:
            _, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        
        if canny_bool:
            img = cv2.Canny(img, 0, 0)
        
        if contour_bool:
            contours, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > area:  # checking if the contour is a dice
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(img_2, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    dice = crop_dice(img, (x, y, w, h))
                    dots = find_dots(dice, param_1, param_2, 0, 0)


                    if dots is not None:
                        dots = dots[dots[:, 2] < max_rad]
                        for dot_x, dot_y, dot_rad in dots:
                            cv2.circle(img_2, (x+dot_x, y+dot_y), dot_rad, (0, 0, 255), 2)
                    
                        num_of_dots = dots.shape[0]
                        cv2.putText(img_2, f'Dots: {num_of_dots}', (int(x), int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


        cv2.imshow('Transformed_img', img)
        cv2.imshow('Final_img', img_2)

        if cv2.waitKey(1) == 1: break
    
    cv2.destroyAllWindows()


def test_images(path='sample_images/'):
    """Tests images in a given directory with given parameters.
    Plots the results.

    Args:
        path (str, optional): the path to the directory with images. Defaults to 'sample_images/'.
    """
    gaussian_kernel = (5, 5)
    gaussian_sigma = 4
    threshold = 120
    contour_area = 1000
    param_1 = 50
    param_2 = 15
    max_rad = 20


    for i, file in enumerate(os.listdir(path), start=1):
        plt.subplot(3, 4, i)
        plt.xticks([])
        plt.yticks([])
        base_img = cv2.imread(f'{path}{file}')
        img_copy = base_img.copy()
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
        img_copy = cv2.GaussianBlur(img_copy, gaussian_kernel, gaussian_sigma)
        _, img_copy = cv2.threshold(img_copy, threshold, 255, cv2.THRESH_BINARY)
        img_copy = cv2.Canny(img_copy, 0, 0)

        contours, _ = cv2.findContours(img_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cont in contours:
            if cv2.contourArea(cont) > contour_area:  # if true then it's a dice
                    (x, y, w, h) = cv2.boundingRect(cont)
                    cv2.rectangle(base_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    dice = crop_dice(img_copy, (x, y, w, h))
                    dots = find_dots(dice, param_1, param_2, 0, max_rad)

                    if dots is not None:
                        dots = dots[dots[:, 2] < max_rad]
                        for dot_x, dot_y, dot_rad in dots:
                            cv2.circle(base_img, (x+dot_x, y+dot_y), dot_rad, (0, 0, 255), 2)
                    
                        num_of_dots = dots.shape[0]
                        cv2.putText(base_img, f'Dots: {num_of_dots}', (int(x), int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        plt.imshow(base_img)
    plt.show()



if __name__ == '__main__':
    # main()
    # plot_transformations()
    test_images(path='sample_images/zoomed/')