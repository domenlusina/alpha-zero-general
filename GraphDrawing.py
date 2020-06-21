import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def display_graph(data, ind, save_path=None, title='Title', display=False, xlabel='Starting heuristic probability [%]',
                  xlim=101, ylim=101, xstep=10, ystep=10, colwidth=0.7):
    data = np.array(data)
    wins = data[:, 0]
    loses = data[:, 1]
    draws = data[:, 2]

    c = ['#396AB1', '#DA7C30', '#3E9651', '#CC2529']  # our color palette
    c1 = []
    c2 = []
    c3 = []
    for i, j in enumerate(ind):
        v = 1.375 - 0.0075 * j
        c1.append(lighten_color(c[0], v))
        c2.append(lighten_color(c[1], v))
        c3.append(lighten_color(c[2], v))

    # drawing stacked bars
    b1 = plt.bar(ind, wins, width=colwidth, color=c1)
    b2 = plt.bar(ind, draws, width=colwidth, bottom=wins, color=c2)
    b3 = plt.bar(ind, loses, width=colwidth, bottom=draws + wins, color=c3)

    # drawing line
    middle = wins + draws / 2
    plt.plot(ind, middle, 'o-', color=c[3], linewidth=1, markersize=2)

    # writings
    plt.ylabel('Percent of games [%]')
    plt.xlabel(xlabel)
    plt.title(title)

    # legend
    leg = plt.legend((b1[0], b2[0], b3[0]), ('Wins', 'Draws', 'Loses'), loc='upper right')
    for i in range(3):
        leg.legendHandles[i].set_color(c[i])

    # limit axis
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)

    # ticks
    plt.xticks(np.arange(0, xlim, step=xstep), rotation=90)
    plt.yticks(np.arange(0, ylim, step=ystep))
    aml = AutoMinorLocator(10)
    plt.axes().xaxis.set_minor_locator(aml)
    plt.axes().yaxis.set_minor_locator(aml)

    if display:
        plt.show()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.clf()
