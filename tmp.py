from matplotlib import pyplot as plt
import pandas as pd


if __name__ == '__main__':
    csv = pd.read_csv('helpme.csv')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.hist(csv.values[:], rwidth=0.8)
    plt.tight_layout()
    plt.savefig("helpme.png")

    # for n in [0, 2, 4, 7, 9, 11]:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #
    #     ax.hist(csv.values[:, n], rwidth=0.8)
    #     plt.tight_layout()
    #     plt.savefig("help_{}.png".format(n))
