import numpy as np
# function to get distance from a point (y) to a line connected by two vertices
# (p1, p2) from stackoverflow.com/questions/849211/
# Consider the line extending the segment, parameterized as v + t (w - v).
# We find projection of point p onto the line.
# It falls where t = [(p-v) . (w-v)] / |w-v|^2


# calculate dot(a) of a(n,2), b(n,2): np.add.reduce(b1 * b2, axis=1)
# calucalte norm(a) of a(n,2), b(n,2): np.add.reduce((a-b)**2, axis=1)
def calc_t(a):
    b1 = a[:, 0, :] - a[:, 1, :]
    b2 = a[:, 2, :] - a[:, 1, :]
    dots = np.add.reduce(b1 * b2, axis=1)
    l2 = np.add.reduce((a[:, 1] - a[:, 2])**2, axis=1)
    return np.clip(dots / l2, 0, 1)


def get_diff(t, a):
    projection = a[:, 1, :] + np.repeat(
        t.reshape(-1, 1), 2, axis=1) * (a[:, 2, :] - a[:, 1, :])
    outside_bounds = (t < 0) ^ (t > 1)
    out = np.add.reduce(
        (a[:, 0, :] - projection) * (a[:, 0, :] - projection),
        axis=1
    )
    end_point_distance = np.amin([
        np.add.reduce((a[outside_bounds, 1] - a[outside_bounds, 0])**2, axis=1),
        np.add.reduce((a[outside_bounds, 2] - a[outside_bounds, 0])**2, axis=1)
    ], axis=0)
    # If we have gone beyond endpoints, set distance to be the distance to the
    # end point (rather than to a continuation of the line)
    out[outside_bounds] = end_point_distance
    return np.min(out)


v_calc_t = np.vectorize(calc_t, signature='(a,b,c)->(a)')
v_get_diff = np.vectorize(get_diff, signature='(a),(a,b,c)->()')


def minimum_distance(a, b):
    # construct our tensor (allowing vectorization)
    # m{i, j, k, p}
    # i iterates over each point in a
    # j cycles through each pair of points in b
    # k cycles through (a[i], b[j], b[j+1])
    # p each of which has [x, y]
    m = np.zeros((a.shape[0], b.shape[0] - 1, 3, 2))
    m[:, :, 0, :] = np.transpose(
        np.tile(a, [m.shape[1] + 1, 1, 1]),
        axes=[1, 0, 2]
    )[:, :-1, :]
    m[:, :, 1, :] = np.tile(b, [a.shape[0], 1, 1])[:, :-1, :]
    m[:, :, 2, :] = np.tile(
        np.roll(b, -1, axis=0), [a.shape[0], 1, 1]
    )[:, :-1, :]
    # t[i, j] = ((a[i] - b[j]) . (b[j + 1] - b[j])) / (b[j + 1] - b[j]|**2
    t = v_calc_t(np.array(m))
    return np.sum(v_get_diff(t, m)) / a.shape[0]


def arc_distance_fast(a, b):
    return (
        minimum_distance(a, b) + minimum_distance(b, a)
    )


def calculate_distance_matrix(cls):
    distances = np.zeros((len(cls), len(cls)))
    for i in range(len(cls)):
        for j in range(i + 1, len(cls)):
            distances[i, j] = arc_distance_fast(cls[i], cls[j])
    distances += np.transpose(distances)
    return distances
