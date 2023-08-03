import numpy as np

window_id = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
mask = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0])


def gen_key():
    return mask.cumsum(0)


def gen_query():
    return (
        np.full(mask.shape, mask.sum()) + 1,
        mask.cumsum(0),
        np.array([5, 5, 5, 5, 7, 7, 7, 7, 6, 6, 6, 6]) + 1,
    )


def compose(q, k):
    output = np.zeros((12, 12))

    for i in range(12):
        for j in range(12):
            gi = i // 4
            gj = j // 4
            if gi == gj:
                output[i, j] = q[1][i] - k[j]
            elif gi < gj:
                output[i, j] = q[0][i] - k[j] + q[1][i] - 1
            else:
                output[i, j] = q[1][i] - k[j]

    output[mask == 0] = 0
    output[:, mask == 0] = 0
    return output


print(compose(gen_query(), gen_key()))
