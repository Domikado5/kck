import numpy as np
import matplotlib.pyplot as plt
import cv2

def null_func(arg):
    pass


def draw_circles(image, circles):
    circles_np = np.round(circles[0, :]).astype('int')
    for (x, y, radius) in circles_np:
        cv2.circle(image, (x, y), radius, (255, 0, 0), 2)


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

    base_img = cv2.imread('dices.jpg')

    while True:
        img = base_img.copy()
        img_2 = base_img.copy()

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
        area            = cv2.getTrackbarPos('contour_area', 'window')

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
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > area:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(img_2, (x, y), (x+w, y+h), (0, 255, 0), 2)


        cv2.imshow('Transformed_img', img)
        cv2.imshow('Final_img', img_2)

        if cv2.waitKey(1) == 1: break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()