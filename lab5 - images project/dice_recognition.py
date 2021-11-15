import numpy as np
import cv2
import matplotlib.pyplot as plt


def null_func(arg):
    pass


def draw_circles(image, circles):
    circles_np = np.round(circles[0, :]).astype('int')
    for (x, y, radius) in circles_np:
        cv2.circle(image, (x, y), radius, (255, 0, 0), 2)


def count_dots(dice, thresh=500):
    """Counts dots of a dice

    Args:
        dice (np.ndarray): a cropped image of a dice
        thresh (int, optional): the area below which a contour will be considered a dot. Defaults to 500.

    Returns:
        int: number of dots
    """
    dice_copy = dice.copy()

    contours, _ = cv2.findContours(dice_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dots = filter(lambda cont: cv2.contourArea(cont) < thresh, contours)

    return len(list(dots))


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
    cv2.namedWindow('window', cv2.WINDOW_NORMAL)

    cv2.createTrackbar('grayscale', 'window', 0, 1, null_func)
    cv2.createTrackbar('gaussian_toggle', 'window', 0, 1, null_func)
    cv2.createTrackbar('kernel_gaussian', 'window', 1, 5, null_func)
    cv2.createTrackbar('sigma_gaussian', 'window', 0, 10, null_func)
    cv2.createTrackbar('threshold_toggle', 'window', 0, 1, null_func)
    cv2.createTrackbar('threshold', 'window', 0, 255, null_func)
    cv2.createTrackbar('canny_toggle', 'window', 0, 1, null_func)
    cv2.createTrackbar('canny_thresh_1', 'window', 0, 255, null_func)
    cv2.createTrackbar('canny_thresh_2', 'window', 0, 255, null_func)
    cv2.createTrackbar('contour_toggle', 'window', 0, 1, null_func)
    cv2.createTrackbar('contour_area', 'window', 0, 1000, null_func)
    cv2.createTrackbar('dot_area', 'window', 0, 1000, null_func)

    base_img = cv2.imread('sample_images/dices.jpg')

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
        canny_thresh_1  = cv2.getTrackbarPos('canny_thresh_1', 'window')
        canny_thresh_2  = cv2.getTrackbarPos('canny_thresh_2', 'window')
        contour_bool    = cv2.getTrackbarPos('contour_toggle', 'window') == 1
        area            = cv2.getTrackbarPos('contour_area', 'window')  # the minimum area of a dice
        dot_area        = cv2.getTrackbarPos('dot_area', 'window')  # the maximum area of a dot

        if gray_bool:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if gaussian_bool:
            if kernel_g % 2 == 1:
                img = cv2.GaussianBlur(img, (kernel_g, kernel_g), sigma_g)
        
        if threshold_bool:
            _, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        
        if canny_bool:
            img = cv2.Canny(img, canny_thresh_1, canny_thresh_2)
        
        if contour_bool:
            contours, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > area:  # checking if the contour is a dice
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(img_2, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    dice = crop_dice(img, (x, y, w, h))
                    num_of_dots = count_dots(dice, dot_area)

                    cv2.putText(img_2, f'Dots: {num_of_dots}', (int(x+w), int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


        cv2.imshow('Transformed_img', img)
        cv2.imshow('Final_img', img_2)

        if cv2.waitKey(1) == 1: break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    # plot_transformations()