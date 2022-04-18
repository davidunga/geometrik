from curves import *
from test_utils import get_shape_points


def test_consistency():
    """
    Test that converting a curve to a different curve, and then back again,
    yields the original curve.
    """

    _err_thresh = 0.05

    xy, _ = get_shape_points('ellipse')
    n = 500
    r = np.linspace(.01, 2, n) ** 2
    t = np.linspace(.01, np.pi - .01, n)
    s = np.linspace(0, 5, n)

    curves = [
        RadiusProfile(t=t, r=r),
        RadiusProfile(t=t[::-1], r=r),
        RadiusProfile(t=t, r=r[::-1]),
        RadiusProfile(t=t[::-1], r=r[::-1]),
        AngleProfile(s=s, t=t),
        AngleProfile(s=s, t=t[::-1]),
        Cartesian(xy=xy - xy[0]),
        Cartesian(xy=np.fliplr(xy - xy[0])),
        Cartesian(xy=np.flipud(xy - xy[-1])),
    ]

    error_reports = []
    for curve in curves:
        for cnvrt_type in [RadiusProfile, AngleProfile, Cartesian]:
            src_type = type(curve)

            curve_converted = cnvrt_type(curve)
            curve_reconstructed = src_type(curve_converted)

            if src_type == Cartesian:
                err = np.max(np.abs(curve.as_np() - curve_reconstructed.as_np()))
            else:
                dt = curve.t - curve_reconstructed.t
                if dt[0] < 0:
                    dt *= -1
                dt -= np.round(dt[0] / (2 * np.pi)) * (2 * np.pi)
                dt_err = np.max(np.abs(dt))
                if src_type == RadiusProfile:
                    err = np.max([dt_err, np.max(np.abs(curve.r - curve_reconstructed.r))])
                else:
                    err = np.max([dt_err, np.max(np.abs(curve.s - curve_reconstructed.s))])

            report = f"{src_type} -> {cnvrt_type} -> {src_type}" + " Error={:2.2f}".format(err)
            print(report)

            if err > _err_thresh:
                error_reports.append(report)

    print("Conversion consistency test done.")
    if len(error_reports) == 0:
        print("No issues were found.")
    else:
        print("Major errors:")
        for error_report in error_reports:
            print(error_report)


if __name__ == "__main__":
    test_consistency()
