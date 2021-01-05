import numpy as np

class Patch:
    # img -> initial image
    # i -> vertical coordinate ranging from 0 to n-1
    # j -> horizontal coordinate ranging from 0 to n-1
    # n -> number of patches in each direction
    # s -> patch size
    def __init__(self, img, i, j, n, s):

        h = img.shape[2]  # Height of initial image
        w = img.shape[3]  # Width of initial image

        x = h // n * (2 * i + 1) // 2  # x Pixel coordinate of patch center
        y = w // n * (2 * j + 1) // 2  # y Pixel coordinate of patch center

        self.top = x - s // 2  # Topmost pixel coordinate
        self.bottom = x + s // 2  # Bottommost pixel coordinate
        self.left = y - s // 2  # Leftmost pixel  coordinate
        self.right = y + s // 2  # Rightmost pixel coordinate

        filled_top, filled_bottom, filled_left, filled_right = 0, s, 0, s
        if self.top < 0:
            filled_top = -self.top
        if self.bottom > h:
            filled_bottom = s - (self.bottom - h)
        if self.left < 0:
            filled_left = -self.left
        if self.right > w:
            filled_right = s - (self.right - w)

        self.patch = np.zeros((1, 1, s, s))
        self.patch[:, :, filled_top:filled_bottom, filled_left:filled_right] = img[:, :,max(self.top, 0):min(self.bottom, h), max(self.left, 0):min(self.right, w)]


def patchify(img, n, patch_size):
    patches = []  # List containing all patches from initial_img
    # Iterate over a nxn grid
    for i in range(n):
        for j in range(n):
            patch = Patch(img, i, j, n, patch_size)  # Create patch
            patches.append(patch)  # Add patch to list
    return patches


def unpatchify(patches, crop, s):
    # UNPATCHIFY
    img_reconstructed = np.zeros((1, 1, 500 + s + 500, 500 + s + 500))
    img_reconstructed_count = np.zeros((1, 1, 500 + s + 500, 500 + s + 500))

    for patch in patches:
        l, r, t, b = patch.left + crop // 4, patch.right - crop // 4, patch.top + crop // 4, patch.bottom - crop // 4
        img_reconstructed[:, :, t + 500:b + 500, l + 500:r + 500] += patch.patch[:, crop // 4:-crop // 4,
                                                                     crop // 4:-crop // 4]
        img_reconstructed_count[:, :, t + 500:b + 500, l + 500:r + 500] += 1

    return img_reconstructed[:, :, 500:500 + s, 500:500 + s] / img_reconstructed_count[:, :, 500:500 + s, 500:500 + s]