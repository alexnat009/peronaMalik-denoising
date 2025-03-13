import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


def display_errors(image, approximation):
    # return mean squared error
    return np.sqrt(np.mean((image - approximation) ** 2))


def show_results(img, initial_time, initial_kappa, initial_alpha, iteration, method, g):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 12), layout=None)  # type:figure.Figure, axes.Axes
    fig.canvas.manager.set_window_title(method.__name__)
    ax = ax.flatten()
    lines = []
    diffusionMethods = list(g.keys())

    # initial drawings
    for iter, choice in enumerate(diffusionMethods):
        denoise = method(g[f"{choice}"], img, delta_t=initial_time,
                         iterations=iteration,
                         kappa=initial_kappa,
                         alpha=initial_alpha)
        lines.append(ax[iter + 1].imshow(denoise, cmap='gray'))
        ax[iter + 1].axis('off')
        ax[iter + 1].set_title(f"{choice}")
    ax[0].imshow(img, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Original Image')

    fig.subplots_adjust(left=0.25, bottom=0.25)

    axKappa = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    kappa_slider = Slider(ax=axKappa, label='Kappa', valmin=0.1, valmax=100, valinit=initial_kappa,
                          orientation="horizontal")

    axAlpha = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    alpha_slider = Slider(ax=axAlpha, label="Alpha", valmin=0.1, valmax=30, valinit=initial_alpha,
                          orientation="vertical")

    axTime = fig.add_axes([0.2, 0.25, 0.0225, 0.63])
    time_slider = Slider(ax=axTime, label="Time", valmin=0.1, valmax=2, valinit=initial_time, orientation="vertical")

    def update(val):
        kappa = kappa_slider.val
        alpha = alpha_slider.val
        time = time_slider.val
        for line, coefficient in zip(lines, diffusionMethods):
            denoised = method(g[f"{coefficient}"], img, delta_t=time, iterations=iteration, kappa=kappa,
                              alpha=alpha)
            error = display_errors(img, denoised)

            print(f'Method Used: {method.__name__}')
            print(f'Diffusion Coefficient: {coefficient}')
            print(f'Parameter Settings: Kappa={kappa}, Alpha={alpha}, Iterations={iteration}, Time={time}')
            print(f'RMSE: {error:.4f}\n')
            line.set_data(denoised)
        fig.canvas.draw_idle()

    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        kappa_slider.reset()
        alpha_slider.reset()
        time_slider.reset()

    button.on_clicked(reset)

    kappa_slider.on_changed(update)
    alpha_slider.on_changed(update)
    time_slider.on_changed(update)
    plt.show()
