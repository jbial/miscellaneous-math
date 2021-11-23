"""Generate a fractal via Newton's Method
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pprint import pprint


def get_poly(coeffs):
    """Gets polynomial object and it's roots with the domain adjusted to include the roots

    Args:
        coeffs (List[int]): list of polynomial coefficients

    Returns:
        (np.polynomial.Polynomial, np.array, float): numpy polynomial + roots + grid width
    """
    poly = np.polynomial.Polynomial(coeffs)
    
    # get roots and suitable bounds on the complex plane
    roots = poly.roots()
    width = np.ceil(np.abs(roots).max())

    return poly, roots, width


def newton_raphson(polynomial, width, gridsize, iters=100, scale=1.0, tol=1e-7):
    """Runs the Newton-Raphson method on the complex plane

    Args:
        polynomial (np.polynomial.Polynomial): polynomial object
        width (float): upper bound of the complex domain
        gridsize (int): number of points along one dimension in the gridsize
        iters (int): maximum number of iterations to run newton-raphson

    Returns: 
        (np.array, np.array): original grid points and iterated grid points

    Note:
        Assumes a square grid of [gridsize x gridsize]
    """
    poly_deriv = polynomial.deriv()

    # generate square grid with gridsize^2 of points in [-width, width]^2 embedded in complex plane
    Y, X = np.mgrid[width:-width:1j*gridsize, -width:width:1j*gridsize]
    grid = (X + 1j * Y) / scale

    num_points = np.prod(grid.shape)
    x = np.copy(grid)
    for i in tqdm(range(iters), desc=f"Running Newton-Raphson for {num_points} points"):
        fx = polynomial(x)
        df_dx = poly_deriv(x)
        x = x - fx / df_dx

        # stopping condition - all points converge to a root
        if (fx < tol).sum() == num_points: 
            print(f"Stopped on iteration {i}")
            print(np.unique(x.flatten(), axis=0))
            break

    return grid, x


def label(roots, solution):
    """Labels points from Newton's method according to the nearest root

    Args:
        roots (np.array): array of the polynomial roots
        solution (np.array): grid of iterated complex points

    Returns:
        np.array: grid of ints corresponding to index of the root
    """
    absdiff = np.abs(solution.reshape(1, *solution.shape) - roots.reshape(-1, 1, 1))
    return np.argmin(absdiff, axis=0)


def color_interp(num_colors, temp=0.1):
    """Gets a smooth linear interpolation between two random RGB colors
    """
    src_color, dst_color = np.random.rand(2, 3)

    # sharpen the color
    src_color = np.exp(src_color / temp) / np.exp(src_color / temp).sum()
    dst_color = np.exp(dst_color / temp) / np.exp(dst_color / temp).sum()

    return [l * src_color + (1 - l) * dst_color for l in np.linspace(0, 1, num_colors)]


def plot_fractal(labels, colormap, filename="fractal.png"):
    """Maps the labels to a predefined colormap and plots the image

    Args:
        labels (np.array): labels grid of nearest root to a given point
        colormap (Dict[int]): map of root index to RGB color
        filename (str): filename to save the image to
    """
    image = np.zeros((*labels.shape, 3))

    # fill colors
    for label in colormap:
        image[labels == label] = colormap[label]

    fig = plt.figure(figsize=(40, 40))
    plt.imshow(image, cmap='inferno')
    plt.axis('off')
    plt.savefig(f"figures/{filename}", dpi=200)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coeffs", 
        type=float, 
        nargs='+', 
        help="Polynomial coefficients e.g. 1 2 3 -> 1 + 2x + 3x^2"
    )
    parser.add_argument(
        "--gridsize", 
        type=int, 
        default=64,
        help="sqrt(# of points in the complex plane) e.g. image width"
    )
    parser.add_argument(
        "--iters", 
        type=int, 
        default=100,
        help="sqrt(# of points in the complex plane) e.g. image width"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="fractal.png",
        help="Filename to save fractal image to"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=99,
        help="Filename to save fractal image to"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale of grid points (zoom factor)"
    )
    args = parser.parse_args()

    if args.coeffs is None:
        np.random.seed(args.random_seed)
        rand_degree = np.random.randint(5, 15)
        rand_coeffs = np.random.normal(scale=10, size=rand_degree)
        print("Randomly generated polynomial (lowest degree to highest):")
        pprint(rand_coeffs.tolist())

    # run newton raphson and produce the fractal image
    polynomial, roots, width = get_poly(args.coeffs or rand_coeffs)
    grid, outputs = newton_raphson(polynomial, width, args.gridsize, args.iters, scale=args.scale)
    labels = label(roots, outputs)

    colors = color_interp(len(roots))

    plot_fractal(
        labels, 
        {i: c  for i, c in enumerate(np.random.rand(len(roots), 3))},
        filename=args.filename
    )


if __name__ == '__main__':
    main()
