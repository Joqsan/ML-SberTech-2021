import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from IPython.display import display, Math


def sample_point(x, eps=1):
    x1 = np.random.uniform(x[0] - eps, x[0] + eps)
    x2 = np.random.uniform(x[1] - eps, x[1] + eps)

    return np.array([x1, x2])


class StochasticGradientDescent:

    def __init__(self, x_0, x_optimal, lr, f, grad_f, eps):
        self.x_list = []
        self.x_curr = x_0
        self.lr = lr
        self.f = f
        self.grad_f = grad_f
        self.x_optimal = x_optimal
        self.f_optimal = f(x_optimal)
        self.eps = eps

        self.convergence_list = [norm(x_0)]
        self.precision_list = [f(x_0)]

    def optimize(self):
        x_prev = self.x_curr
        x_sampled = sample_point(self.x_curr, self.eps)
        self.x_curr = x_prev - self.lr * self.grad_f(x_sampled)

        self.x_list.append(self.x_curr)
        self.convergence_list.append(norm(self.x_optimal - self.x_curr))
        self.precision_list.append(abs(self.f_optimal - self.f(self.x_curr)))

    def plot_location(self):
        x_array = np.array(self.x_list)
        plt.plot(x_array[:, 0], x_array[:, 1], 'go--', linewidth=2, markersize=10, alpha=0.5, label='SGD')

    def plot_convergence(self):
        plt.plot(self.convergence_list, 'go-', linewidth=2, markersize=10, alpha=0.5, label='SGD')

    def plot_precision(self):
        plt.plot(self.precision_list, 'go-', linewidth=2, markersize=10, alpha=0.5, label='SGD')

    def get_optimal_point(self):
        #self.x_curr = np.array(self.x_list).mean(axis=0)
        return display(Math(r'\hat x\ with\ SGD: ({:.3f},{:.3f})'.format(self.x_curr[0], self.x_curr[1])))

    def get_approx_f_optimal(self):
        return display(Math(r'f(\hat x)\ with\ SGD: {:.3f}'.format(self.f(self.x_curr))))


class MomentumGradientDescent:

    def __init__(self, x_0, x_optimal, lr, f, grad_f, beta=0.5):
        self.x_list = []
        self.x_curr = x_0
        self.lr = lr
        self.beta = beta
        self.f = f
        self.grad_f = grad_f
        self.x_optimal = x_optimal
        self.f_optimal = f(x_optimal)
        self.v_curr = 0

        self.convergence_list = [norm(x_0)]
        self.precision_list = [f(x_0)]

    def optimize(self):
        prev_v = self.v_curr
        x_prev = self.x_curr
        self.v_curr = self.beta * prev_v + self.lr * self.grad_f(x_prev)
        self.x_curr = x_prev - self.v_curr

        self.x_list.append(self.x_curr)
        self.convergence_list.append(norm(self.x_optimal - self.x_curr))
        self.precision_list.append(abs(self.f_optimal - self.f(self.x_curr)))

    def plot_location(self):
        x_array = np.array(self.x_list)
        plt.plot(x_array[:, 0], x_array[:, 1], 'bo--', linewidth=2, markersize=10, alpha=0.5, label='Momentum')

    def plot_convergence(self):
        plt.plot(self.convergence_list, 'bo-', linewidth=2, markersize=10, alpha=0.5, label='Momentum')

    def plot_precision(self):
        plt.plot(self.precision_list, 'bo-', linewidth=2, markersize=10, alpha=0.5, label='Momentum')

    def get_optimal_point(self):
        return display(Math(r'\hat x\ with\ Momentum: ({:.3f},{:.3f})'.format(self.x_curr[0], self.x_curr[1])))

    def get_approx_f_optimal(self):
        return display(Math(r'f(\hat x)\ with\ Momentum: {:.3f}'.format(self.f(self.x_curr))))


class GradientDescent:

    def __init__(self, x_0, x_optimal, lr, f, grad_f):
        self.x_list = []
        self.x_curr = x_0
        self.lr = lr
        self.f = f
        self.grad_f = grad_f
        self.x_optimal = x_optimal
        self.f_optimal = f(x_optimal)

        self.convergence_list = [norm(x_0)]
        self.precision_list = [f(x_0)]

    def optimize(self):
        x_prev = self.x_curr
        self.x_curr = x_prev - self.lr * self.grad_f(x_prev)

        self.x_list.append(self.x_curr)
        self.convergence_list.append(norm(self.x_optimal - self.x_curr))
        self.precision_list.append(abs(self.f_optimal - self.f(self.x_curr)))

    def plot_location(self):
        x_array = np.array(self.x_list)
        plt.plot(x_array[:, 0], x_array[:, 1], 'ro--', linewidth=2, markersize=10, alpha=0.5, label='GD')

    def plot_convergence(self):
        plt.plot(self.convergence_list, 'ro-', linewidth=2, markersize=10, alpha=0.5, label='GD')

    def plot_precision(self):
        plt.plot(self.precision_list, 'ro-', linewidth=2, markersize=10, alpha=0.5, label='GD')

    def get_optimal_point(self):
        return display(Math(r'\hat x\ with\ GD: ({:.3f},{:.3f})'.format(self.x_curr[0], self.x_curr[1])))

    def get_approx_f_optimal(self):
        return display(Math(r'f(\hat x)\ with\ GD: {:.3f}'.format(self.f(self.x_curr))))