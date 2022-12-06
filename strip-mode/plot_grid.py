import matplotlib.pyplot as plt
import numpy as np

def plot(out):
    pix = [prop for prop in out.split(" ") if prop.startswith("pixel")]
    strip_hh = [prop for prop in out.split(" ") if prop.startswith("strip_h")]
    strip_vv = [prop for prop in out.split(" ") if prop.startswith("strip_v")]

    x = np.array([int(p.split(",")[0][7:]) for p in pix])
    y = np.array([int(p.split(",")[1][:-2]) for p in pix])
    n_x, n_y = max(x), max(y)
    n_x = n_x + 2 if n_x % 2 == 0 else n_x + 1
    n_y = n_y + 2 if n_y % 2 == 0 else n_y + 1

    strip_hh = np.array([int(strip.split("(")[1][:-1])-1 for strip in strip_hh])
    strip_vv = np.array([int(strip.split("(")[1][:-1])-1 for strip in strip_vv])

    strip_h = 2 * strip_hh
    strip_v = 2 * strip_vv

    z = np.zeros((n_x,n_y))
    z[x,y] = 1
    z *= -1

    plt.figure(figsize=(12,7))

    plt.subplot(1,1,1)
    plt.imshow(z, cmap='gray', interpolation='nearest')
    plt.title("Grid")
    # Grid plt
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, n_y-1, 2), minor=False)
    ax.set_yticks(np.arange(0.5, n_x-1, 2), minor=False)
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.tick_params(axis='both', which='major', labelsize=0)

    # -1, -1 shifted grid.
    # z = np.zeros((n,n))
    # z[(x+1)%n,(y+1)%n] = 0.5
    # z *= -1

    a = ax.get_xgridlines()
    print(strip_vv, strip_hh)
    print(len(a))
    for i in strip_vv:
        a[i].set_color('b')
    b = ax.get_ygridlines()
    print(len(b))
    for i in strip_hh:
        b[i].set_color('r')

    # if len(strip_h) > 0:
    #     strip_h = np.hstack((strip_h, strip_h+1))
    #     z[strip_h, :] *= 2
    # if len(strip_v) > 0:
    #     strip_v = np.hstack((strip_v, strip_v+1))
    #     z[:, strip_v] *= 2
    # z = np.clip(z, -1, 1)

    # plt.subplot(1,2,2)
    # plt.imshow(z, cmap='gray', interpolation='nearest')
    # plt.title("Tiles (shifted)")
    # ax = plt.gca()
    # ax.set_xticks(np.arange(1.5, n-1, 2), minor=False)
    # ax.set_yticks(np.arange(1.5, n-1, 2), minor=False)
    # ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    # a = ax.get_xgridlines()
    # print(strip_vv, strip_hh)
    # for i in strip_vv:
    #     a[i].set_color('b')
    # b = ax.get_ygridlines()
    # for i in strip_hh:
    #     b[i].set_color('r')

        # a[i].set_linewidth(4)
    # ax.tick_params(axis='both', which='major', labelsize=0)
    plt.show()