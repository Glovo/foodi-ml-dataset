import cv2


def dhash(image, hashSize=64):
    """
    Gets the hash of an image. Converts the image to grayscale and resizes
    to hashSize and computes the hash based on image gradient.
    Parameters
    ----------
    image: np.array (OpenCV's format)
    hashSize : int
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    # compute the relative horizontal gradient between adjacent column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash and return it
    return hash(tuple(diff.flatten()))