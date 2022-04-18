from huh_curves import HuhCurve, HuhParams
import matplotlib.pyplot as plt


def _plot(ax, h: HuhCurve, color):
    plt.sca(ax)
    X = h.full_period_curve(0.001).xy()
    plt.plot(X[:, 0], X[:, 1], color)
    plt.axis('square')


def demo():
    # pure shapes:
    h_a = HuhCurve(HuhParams(m=3, n=2, eps=1.0))
    h_b = HuhCurve(HuhParams(m=3, n=1, eps=1.2))
    h_c = HuhCurve(HuhParams(m=6, n=1, eps=1.6))

    # new shapes from pure shapes:
    h_ab = h_a + h_b
    h_ac = h_a + h_c

    # show:
    fig, axs = plt.subplots(ncols=3, nrows=2)
    fig.set_size_inches(10, 7)

    # --
    _plot(axs[0, 0], h_a, 'r')
    p = h_a.params[0]
    plt.title(f"[A] m={p.m}, n={p.n} eps={p.eps}")

    _plot(axs[0, 1], h_b, 'r')
    p = h_b.params[0]
    plt.title(f"[B] m={p.m}, n={p.n} eps={p.eps}")

    _plot(axs[0, 2], h_ab, 'r')
    plt.title(f"[A+B]")
    # --
    _plot(axs[1, 0], h_a, 'g')
    p = h_a.params[0]
    plt.title(f"[A] m={p.m}, n={p.n} eps={p.eps}")

    _plot(axs[1, 1], h_c, 'g')
    p = h_c.params[0]
    plt.title(f"[C] m={p.m}, n={p.n} eps={p.eps}")

    _plot(axs[1, 2], h_ac, 'g')
    plt.title(f"[A+C]")
    # --
    plt.show()


def test():

    h = HuhCurve([HuhParams(m=5, n=1, eps=1.8),
                  HuhParams(m=5, n=3, eps=1.2)])
    curve = h.full_period_curve(0.01).xy()
    hr = HuhCurve.fromCruve(curve)

    print("Testing reconstruction of HuhCurve from xy samples.")
    print("Original curve:")
    for p in h.params:
        print(" ", p)
    print("Reconstructed curve:")
    for p in hr.params:
        print(" ", p)

    if hr.isclose(h):
        print("SUCCESSFUL reconstruction.")
    else:
        print("FAILED reconstruction.")

    _, axs = plt.subplots(ncols=2, nrows=1)
    _plot(axs[0], h, 'r')
    axs[0].set_title("Original")
    _plot(axs[1], hr, 'g')
    axs[1].set_title("Reconstructed")
    plt.show()


if __name__ == "__main__":
    test()
    demo()
