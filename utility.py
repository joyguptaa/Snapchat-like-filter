import cv2 as cv

def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # iter stands for interpolation, useful in image resize
    # initialize the dimensions & grab the image size


    dim = None
    (h, w) = image.shape[:2] # First 2 channels are of only use


    if width is None and height is None:
        # Return image as it is
        return image

    
    if width is None:
        # Calculate Compression ratio
        r = height / float(h)
        dim = (int(w * r), height)

    
    else:
        # Compression ratio
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized