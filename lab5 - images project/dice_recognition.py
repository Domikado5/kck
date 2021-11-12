import numpy as np
import matplotlib.pyplot as plt
import cv2

def null_func(arg):
    pass


def draw_circles(image, circles):
    circles_np = np.uint16(np.around(circles))
    for i in circles_np[0,:]:
        cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 2)


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
    cv2.createTrackbar('hough_toggle', 'window', 0, 1, null_func)
    cv2.createTrackbar('hough_dp', 'window', 1, 5, null_func)
    cv2.createTrackbar('hough_min_dist', 'window', 0, 50, null_func)

    img = cv2.imread('img.jpg')

    gray_bool       = cv2.getTrackbarPos('grayscale', 'window') == 1
    gaussian_bool   = cv2.getTrackbarPos('gaussian_toggle', 'window') == 1
    kernel_g        = int(cv2.getTrackbarPos('kernel_gaussian', 'window'))
    sigma_g         = cv2.getTrackbarPos('sigma_gaussian', 'window')
    threshold_bool  = cv2.getTrackbarPos('threshold_toggle', 'window') == 1
    thresh          = cv2.getTrackbarPos('threshold', 'window')
    canny_bool      = cv2.getTrackbarPos('canny_toggle', 'window') == 1
    canny_thresh_1  = cv2.getTrackbarPos('canny_thresh_1', 'window')
    canny_thresh_2  = cv2.getTrackbarPos('canny_thresh_2', 'window')
    hough_bool      = cv2.getTrackbarPos('hough_toggle', 'window') == 1
    hough_dp        = cv2.getTrackbarPos('hough_dp', 'window')
    hough_m_dist    = int(cv2.getTrackbarPos('hough_min_dist', 'window'))

    if gray_bool:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if gaussian_bool:
        img = cv2.GaussianBlur(img, (kernel_g, kernel_g), sigma_g)
    
    if threshold_bool:
        img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    
    if canny_bool:
        img = cv2.Canny(img, canny_thresh_1, canny_thresh_2)

    # contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    if hough_bool:
        circles = cv2.HoughCircles(img, cv.HOUGH_GRADIENT, hough_dp, hough_m_dist)
        draw_circles(img, circles)

    cv2.imshow(img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()