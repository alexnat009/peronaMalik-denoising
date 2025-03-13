import numpy as np
import cv2 as cv
from outputImages import show_results
from numericalPeronaMalik import add_noise, perona_malik_with_adams_bashforth, \
    perona_malik_with_explicit_runge_kutta, \
    perona_malik_with_adams_moulton

g = {
    "PeronaMalik": lambda s, k, alpha=1.0: 1 / (1 + (s / k) ** 2),
    "Charbonnier": lambda s, k, alpha=1.0: np.exp(-s ** 2 / k ** 2),
    "Weickert": lambda s, k, alpha=1.0: np.power(1 + s ** 2 / k ** 2, -1 / 2),
    "GeneralPeronaMalik": lambda s, k, alpha=1.0: 1 / (1 + (s / k) ** alpha),
    "YuleNielsen": lambda s, k, alpha=1.0: (1 - np.exp(-s ** 2 / k ** 2)) / (s / k) ** 2
}

print("Given approximation methods aren't stable for large time steps")
image = cv.imread("images/noisy.png", 0)
image = cv.resize(image, (200, 200)).astype(np.float64)
# image = add_noise(image, 'gaussian', 0.05)

show_results(image, initial_time=0.3, initial_alpha=0.1, initial_kappa=0.1, iteration=10,
             method=perona_malik_with_explicit_runge_kutta, g=g)
