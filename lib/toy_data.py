"""
get toy data set
"""
import numpy as np
import sklearn
import sklearn.datasets


def inf_train_gen(data, rng=None, batch_size=200):
    """
    return the standard generated data
    """
    if rng is None:
        rng = np.random.RandomState()

    if data == "2spirals_1d":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n
        d1y = np.sin(n) * n
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        return x

    elif data == "2spirals_2d":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "swissroll_1d":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=0.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data
    
    elif data == "swissroll_2d":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles_1d":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.0)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "circles_2d":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data =="2sines_1d":
        x = (rng.rand(batch_size) -0.5) * 2 * np.pi
        u = (rng.binomial(1,0.5,batch_size) - 0.5) * 2
        y = u * np.sin(x) * 2.5
        return np.stack((x, y), 1)

    elif data =="target_1d":
        shapes = np.random.randint(7, size=batch_size)
        mask = []
        for i in range(7):
            mask.append((shapes==i)*1.)

        theta = np.linspace(0, 2 * np.pi, batch_size, endpoint=False)
        x = (mask[0] + mask[1] + mask[2]) * (rng.rand(batch_size) -0.5) * 4 +\
         (-mask[3] + mask[4]*0.0 + mask[5]) * 2 * np.ones(batch_size) +\
         mask[6] * np.cos(theta)

        y = (mask[3] + mask[4] + mask[5]) * (rng.rand(batch_size) -0.5) * 4 +\
         (-mask[0] + mask[1]*0.0 + mask[2]) * 2 * np.ones(batch_size) +\
         mask[6] * np.sin(theta)

        return np.stack((x, y), 1)

    else:
        return inf_train_gen("2spirals_1d", rng, batch_size)


def generate_slope(n: int = 100, s=0.5, r=5, f=1, random=True, save_path=None, samples=None):
    """
    n: n the number of the scatters, must be int
    r is the range of slope
    s is the slope factor
    f is how flat the slope is. 1 is flat, >1 is not.
    save_path is data/slope_random as default
    """
    if samples is None:
        one_side = int(np.sqrt(n))
        if random:
            X = r * np.random.rand(n)
            Y = r * np.random.rand(n)
        else:
            inter = 2 * r / one_side
            xx = np.arange(-r, r, inter)
            yy = np.arange(-r, r, inter)
            X, Y = np.meshgrid(xx, yy)
        X = X.reshape(n)
        Y = Y.reshape(n)
    else:
        X = samples[:, 0]
        Y = samples[:, 1]

    Z = np.sin(f * X) + np.cos(f * Y)
    Z = Z + np.tan(s) * Y

    data = np.stack((X, Y, Z), axis=1)
    if save_path:
        np.savetxt(fname=save_path, X=data)
    return data


def generate_swissroll3d(n_samples=100, *, noise=0.0, save_path=None):
    """Generate a swiss roll dataset.
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html
    ----------
    n_samples : int, default=100
        The number of sample points on the S curve.
    noise : float, default=0.0
        The standard deviation of the gaussian noise.
    """
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
    x = t * np.cos(t) * 0.5 - 1
    y = 10 * np.random.rand(1, n_samples) - 5.5
    z = t * np.sin(t) * 0.4
    # x = t * np.cos(t)
    # y = 10 * np.random.rand(1, n_samples)
    # z = t * np.sin(t)

    X = np.concatenate((x, y, z, t))
    X[0:3, :] += noise * np.random.rand(3, n_samples)
    data = X.T

    if save_path:
        np.savetxt(fname=save_path, X=data)
    return data
