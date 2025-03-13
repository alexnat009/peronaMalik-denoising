import cv2 as cv
import numpy as np


def add_noise(image, noise_type, noise_percentage):
    match noise_type:
        case "gaussian":
            noise = np.random.normal(0, 1, image.shape)
            noisy_image = image + noise
            return noisy_image
        case "salt_and_pepper":
            img_size = image.size
            noise_size = int(noise_percentage * img_size)
            random_indices = np.random.choice(img_size, noise_size)
            noisy_image = image.copy()
            noise = np.random.choice([np.min(image), np.max(image)], noise_size)
            noisy_image.flat[random_indices] = noise
            return noisy_image


def calculate_gradient(img):
    grad_x, grad_y = np.gradient(img)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return magnitude, grad_x, grad_y


def calculate_initial_s(c, img, delta_t=0.1, number_of_initial_values=4, kappa=1.0, alpha=1.0):
    return np.array([perona_malik_with_explicit_runge_kutta(c, img, delta_t, i, kappa, alpha) for i in
                     range(number_of_initial_values)])


def perona_malik_with_explicit_runge_kutta(c, image, delta_t=0.1, iterations=100, kappa=1.0, alpha=1.0):
    smoothed_image = image.copy()

    grad_x, grad_y = np.gradient(image)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    diffusivity = c(magnitude, kappa, alpha)
    for _ in range(iterations):
        k1 = delta_t * np.gradient(diffusivity * grad_x)[0]
        k2 = delta_t * np.gradient(diffusivity * grad_y)[1]

        k1x = smoothed_image + k1 / 2
        k2y = smoothed_image + k2 / 2

        k3 = delta_t * np.gradient(diffusivity * k1x)[0]
        k4 = delta_t * np.gradient(diffusivity * k2y)[1]

        smoothed_image += (k1 + 2 * k3 + 2 * k4 + k2) / 6

    return smoothed_image


def calculate_adams_gradient(s, s_minus_1, s_minus_2, s_minus_3, f, diff):
    gradX = 55 * f(s, diff)[1] - 59 * f(s_minus_1, diff)[1] + 37 * f(s_minus_2, diff)[1] - 9 * f(s_minus_3, diff)[1]
    gradY = 55 * f(s, diff)[2] - 59 * f(s_minus_1, diff)[2] + 37 * f(s_minus_2, diff)[2] - 9 * f(s_minus_3, diff)[2]
    grad = cv.addWeighted(gradX, 0.5, gradY, 0.5, 0.0)
    return grad


def get_initial_conditions(image, c, delta_t, kappa, alpha, iterations, initial_values=4):
    smoothed_image = image.copy()
    intermediate_results = np.zeros((iterations, image.shape[0], image.shape[1]))

    intermediate_results[0:initial_values] = calculate_initial_s(c, image, delta_t, initial_values, kappa, alpha)
    return smoothed_image, intermediate_results


def get_n_terms(mat, n, i):
    return [mat[n - j] for j in range(i)]


def perona_malik_with_adams_bashforth(c, image, delta_t=0.1, iterations=100, kappa=1.0, alpha=1.0):
    smoothed_image, intermediate_results = get_initial_conditions(image, c, delta_t, kappa, alpha, iterations,
                                                                  initial_values=4)

    f = lambda s, diff: diff * calculate_gradient(s)
    for i in range(3, iterations - 1):
        magnitude, grad_x, grad_y = calculate_gradient(smoothed_image)
        diffusivity = c(magnitude, kappa, alpha)

        s, s_minus_1, s_minus_2, s_minus_3 = get_n_terms(intermediate_results, i, 4)

        grad = calculate_adams_gradient(s, s_minus_1, s_minus_2, s_minus_3, f, diffusivity)
        next_image = s + delta_t / 24 * grad

        smoothed_image = next_image.copy()
        intermediate_results[i + 1] = smoothed_image

    return smoothed_image


def perona_malik_with_adams_moulton(c, image, delta_t=0.1, iterations=100, kappa=1.0, alpha=1.0):
    smoothed_image, intermediate_results = get_initial_conditions(image, c, delta_t, kappa=kappa, alpha=alpha,
                                                                  iterations=iterations, initial_values=4)

    f = lambda s, diff: diff * calculate_gradient(s)
    for i in range(3, iterations - 1):
        magnitude, grad_x, grad_y = calculate_gradient(smoothed_image)
        diffusivity = c(magnitude, kappa, alpha)

        s, s_minus_1, s_minus_2, s_minus_3 = get_n_terms(intermediate_results, i, 4)

        # Predictor step (Adams-Bashforth)
        grad = calculate_adams_gradient(s, s_minus_1, s_minus_2, s_minus_3, f, diffusivity)
        next_image_predict = s + delta_t / 24 * grad

        # Corrector step (Adams-Moulton)
        grad_correct = calculate_adams_gradient(next_image_predict, s, s_minus_1, s_minus_2, f, diffusivity)
        next_image_correct = s + delta_t / 24 * grad_correct

        smoothed_image = next_image_correct.copy()
        intermediate_results[i + 1] = smoothed_image

    return smoothed_image
