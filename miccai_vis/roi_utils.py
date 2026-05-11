def crop_roi(img, center=None, size=96):

    h, w = img.shape

    if center is None:
        center = (h // 2, w // 2)

    y, x = center

    half = size // 2

    return img[y-half:y+half, x-half:x+half]
