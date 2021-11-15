import os
import cv2


def resize_images(path='sample_images/', shape=(512, 384)):
    """Resizes images in a given directory

    Args:
        path (str, optional): the path to the directory with images. Defaults to 'sample_images/'.
        shape (tuple, optional): the desired shape. Defaults to (512, 384).
    """
    (x, y) = shape

    for file in os.listdir(path):
        file_path = f'{path}{file}'
        img = cv2.imread(file_path)
        if img.shape != (y, x, 3):
            img = cv2.resize(img, (x, y), interpolation=cv2.INTER_AREA)
            cv2.imwrite(file_path, img)
            print(f'File {file} has been resized.')
    print('Finished.')


if __name__ == '__main__':
    resize_images()