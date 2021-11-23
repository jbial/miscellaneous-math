import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from itertools import accumulate
from mpl_toolkits import mplot3d


class LorenzAttractor:
    """Classic 3D Lorenz attractor ODE
    """

    def __init__(self, coeffs):
        """
        Args:
            coeffs (Iterable[float]): 3 Lorenz attractor coefficients
        """
        assert len(coeffs) == 3, "Only 3 Lorenz coefficients allowed"
        self.sigma, self.rho, self.beta = coeffs

    def __call__(self, xyz, t):
        return np.array([
            self.sigma * (xyz[1] - xyz[0]),
            xyz[0] * (self.rho - xyz[2]),
            xyz[0] * xyz[1] - self.beta * xyz[2]
        ])


def ode45(ode_fn, init_cond, start, stop, step_size=1e-4):
    """Implements 6th order Runge-Kutta

    Args:
        ode_fn (Callable): implements the first order ODE x' = f(x, t)
        init_cond (Iterable, float): initial value of the ODE
        start (float): integration start time
        end (float): integration end time
        step_size (float): integration step size (dt)

    Returns:
        (np.ndarray, np.ndarray): array of time values and array of integrated values
            both with length int((stop - start) / step_size)
    """
    if isinstance(init_cond, np.ndarray):
        dim = init_cond.shape[0]
    elif isinstance(init_cond, (list, tuple)):
        dim = len(init_cond)
    else:
        dim = 1

    # initialize the integrator params
    times = np.arange(start, stop, step_size)
    values = np.zeros((len(times), dim)).squeeze()
    values[0] = init_cond

    # ode45 (RK(4, 5)) step
    def _step(i, _):
        x, t = values[i], times[i]
        k1 = ode_fn(x, t)
        k2 = ode_fn(
            x + 0.5 * step_size * k1,
            t + 0.5 * step_size
        )
        k3 = ode_fn(
            x + 0.5 * step_size * k2, 
            t + 0.5 * step_size
        )
        k4 = ode_fn(x + step_size * k3, t + step_size)

        values[i + 1] = x + (step_size / 6) * (k1 + 2 * (k2 + k3) + k4)

        return i + 1

    # populate the array of values
    _ = list(accumulate(range(len(times)), _step))

    return times, values


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t0', '--start', type=float, default=2.5, help="Time boundary (start)")
    parser.add_argument('-tn', '--stop', type=float, default=40., help="Time boundary (finish)")
    parser.add_argument('-dt', '--step', type=float, default=1e-4, help="Integrator step size (dt)")
    args = parser.parse_args()

    # ode init stuff
    init_cond = np.array([100, -1., 9.])
    coeffs = [10., 100., 8/3]
    ode = LorenzAttractor(coeffs)

    start_time = time.time()
    times, values = ode45(ode, init_cond, args.start, args.stop, args.step)
    print(
        f"""Time range [{args.start}, {args.stop}] (seconds) | Duration: {time.time() - start_time:.3f} seconds"""
        f""" | Precision: {args.step} | Trajectory length: {len(values)} points"""
    )

    ax = plt.axes(projection='3d')
    coeff_labels = ', '.join(f'{k}={v:.2f}' for k, v in zip([r'$\sigma$', r'$\rho$', r'$\beta$'], coeffs))
    ivp_labels = ', '.join(f'{k}={v:.2f}' for k, v in zip([r'$x_0$', r'$y_0$', r'$z_0$'], init_cond))
    ax.set_title(f"Lorenz Attractor ({coeff_labels})")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.plot(*values.T, color='b', linewidth=0.5)
    ax.scatter(*init_cond, marker='*', color='r', label=ivp_labels)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()