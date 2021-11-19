import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def get_dots(img, pos, min_dot_size=100, max_dot_size=1000):
    """Finds dots of a given dice

    Args:
        img (np.ndarray): image
        pos (tuple): a tuple containing the position of the dice
        min_dot_size (int): the minimum area of a dot
        max_dot_size (int): the maximum area of a dot

    Returns:
        int: number of dots
        list: a list containing the positions of the dots
    """
    (x, y, w, h) = pos

    dice = img[y:y+h, x:x+w]

    dots, hierarchy = cv2.findContours(dice, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    filtered_dots = []
    perimeters = []
    areas = []

    for dot in dots:
        if min_dot_size < cv2.contourArea(dot) < max_dot_size:
            filtered_dots.append(dot)
            perimeter = cv2.arcLength(dot, True)
            perimeters.append(perimeter)
            areas.append(cv2.contourArea(dot))
    filtered_dots = [dot for idx, dot in enumerate(filtered_dots)\
         if perimeters[idx] > max(perimeters)*0.65 and \
             areas[idx] > max(areas)*0.5]
    print(perimeters)
    print(areas)
    print('-----')

    return len(filtered_dots), filtered_dots


def all_transformations(img_path='images/easy/1.jpg', min_area=2000, max_area=100000):
    """Performs all transformations and contours dice.

    Args:
        img_path (str, optional): the path to the image. Defaults to 'images/easy/1.jpg'.
        min_area (int, optional): the min area value to filter contours. Defaults to 2000.
        max_area (int, optional): the max area value to filter contours. Defaults to 100000.

    Returns:
        np.ndarray: image after transformations
    """
    img = cv2.imread(img_path)
    img_copy = img.copy()

    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    img_copy = cv2.medianBlur(img_copy, 5)
    thresh = np.quantile(img_copy, 0.8)
    img_copy = cv2.filter2D(img_copy, -1, sharpen_kernel)
    _, img_copy = cv2.threshold(img_copy, thresh, 255, cv2.THRESH_BINARY_INV)
    img_copy = cv2.morphologyEx(img_copy, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
    contours, _ = cv2.findContours(img_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    dots_total = 0

    for contour in contours:
        if min_area < cv2.contourArea(contour) < max_area:
            (x, y, w, h) = cv2.boundingRect(contour)
            
            num_of_dots, dots = get_dots(img_copy, (x, y, w, h))

            for dot in dots:
                x_d, y_d, w_d, h_d = cv2.boundingRect(dot)
                cv2.ellipse(img, (int(x+x_d+w_d/2), int(y+y_d+h_d/2)), (int(w_d/2), int(h_d/2)), 0, 0, 360, (255, 0, 0), 2)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f'Dots: {num_of_dots}', (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            dots_total += num_of_dots
    
    cv2.putText(img, f'Total: {dots_total}', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return img


def show_transformations(img_path='images/easy/1.jpg', min_area=2000, max_area=100000):
    """Shows the effect every transformation on a picture

    Args:
        img_path (str, optional): the path to the image. Defaults to 'images/easy/1.jpg'.
        min_area (int, optional): the min_area value. Defaults to 2000.
        max_area (int, optional): the max area value. Defaults to 100000.
    """
    img = cv2.imread(img_path)
    img_copy = img.copy()

    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    thresh_val = np.quantile(gray, 0.8)
    cv2.imshow(img_path + '-gray', gray)

    blur = cv2.medianBlur(gray, 5)
    cv2.imshow(img_path + '-blur', blur)

    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    cv2.imshow(img_path + '-sharpen', sharpen)

    _, thresh = cv2.threshold(sharpen, thresh_val, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow(img_path + '-threshold', thresh)

    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
    cv2.imshow(img_path + '-closing', close)

    contours, _ = cv2.findContours(close, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if min_area < cv2.contourArea(contour) < max_area:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow(img_path + '-contours', img)


def test_all(path='images/hard/', min_area=2000, max_area=100000):
    """Tests all images in a given directory
    Args:
        path (str, optional): the path to the directory with images. Defaults to 'images/easy/'.
        min_area (int, optional): the min_area value. Defaults to 2000.
        max_area (int, optional): the max area value. Defaults to 100000.
    """
    for image in os.listdir(path):
        img_path = f'{path}{image}'
        # if image not in ['4.jpg']:
        #     continue
        cv2.imshow(image, all_transformations(img_path, min_area=min_area, max_area=max_area))

        # show_transformations(img_path=img_path)

        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    test_all()