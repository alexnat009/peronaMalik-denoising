
# Image Denoising using Numerical Methods for PDEs

This repository contains a Python project focused on image denoising using various numerical methods for solving partial differential equations (PDEs). The project leverages the Perona-Malik anisotropic diffusion model and implements several numerical methods, including Runge-Kutta, Adams-Bashforth, and Adams-Moulton, to solve the PDEs.

## Project Structure

- `main.py`: Main script to run the image denoising process.
- `numericalODE.py`: Contains implementations of numerical methods for solving ordinary differential equations (ODEs).
- `numericalPeronaMalik.py`: Implements the Perona-Malik model with different numerical methods.
- `peronaMalik.py`: Demonstrates the use of various numerical methods on a sample ODE.
- `outputImages.py`: Provides functions to display the results and visualize the denoised images.

## Dependencies

The project uses the following libraries:
- `numpy` and `scipy`: For numerical computations.
- `opencv-python`: For image processing.
- `matplotlib`: For visualization.

You can install the required dependencies using:
```bash
pip install numpy opencv-python matplotlib scipy
```

## Usage

1. **Run the main script**:
   ```bash
   python main.py
   ```

2. **Add noise to an image**:
   The `add_noise` function in `numericalPeronaMalik.py` can be used to add Gaussian or salt-and-pepper noise to an image.

3. **Denoise an image**:
   The `show_results` function in `outputImages.py` displays the original and denoised images using different numerical methods.

### Output sample


<img src="https://github.com/user-attachments/assets/106d6779-8deb-4721-a2de-6ba4d326a30b" width="1280" height="720">

## Numerical Methods Implemented

- **Runge-Kutta**: A method for solving ODEs.
- **Adams-Bashforth**: A multistep method for solving ODEs.
- **Adams-Moulton**: An implicit method for solving ODEs.
- **Three-Stage Diagonally Implicit Runge-Kutta**: A method for solving stiff ODEs.

## Example

The `peronaMalik.py` file demonstrates the use of various numerical methods on a sample ODE. It plots the approximate and exact solutions for comparison.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
