import matplotlib.pyplot as plt


def bar_plot(x, y, xlabel=None, ylabel=None, title=None, degrees=None, color='maroon'):
    plt.bar(x, y, color=color, width=0.4)
    if degrees:
        plt.xticks(rotation=degrees)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"plots/{title}")
    plt.show()


def line_plot(df, title):
    df.plot()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"plots/{title}")
    plt.show()
