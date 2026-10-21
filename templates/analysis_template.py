"""
قالب استاندارد تحلیل داده — کپی کنید به expXX/analysis.py و ویرایش کنید.
اجرا:  python analysis.py            (داده واقعی از data/run1.csv)
       python analysis.py --demo     (داده مصنوعی برای تست خط‌لوله تحلیل)
وابستگی‌ها: numpy, scipy, pandas, matplotlib   (pip install numpy scipy pandas matplotlib)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
(HERE / "figures").mkdir(exist_ok=True)


def load(demo: bool) -> pd.DataFrame:
    """ستون‌ها: x, y, sy  (sy = عدم‌قطعیت استاندارد y؛ یکای هر ستون در نام آن بیاید)."""
    if demo:
        rng = np.random.default_rng(1)
        x = np.linspace(1, 10, 10)
        sy = np.full_like(x, 0.15)
        y = 2.0 * x + 0.5 + rng.normal(0, sy)
        return pd.DataFrame({"x": x, "y": y, "sy": sy})
    return pd.read_csv(HERE / "data" / "run1.csv")


def model(x, a, b):
    """مدل خطی y = a x + b ؛ برای آزمایش خودتان تغییر دهید (مثلاً نمایی برای RC)."""
    return a * x + b


def main(demo: bool = False) -> None:
    df = load(demo)
    x, y, sy = df["x"].to_numpy(), df["y"].to_numpy(), df["sy"].to_numpy()

    popt, pcov = curve_fit(model, x, y, sigma=sy, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))

    resid = y - model(x, *popt)
    chi2 = np.sum((resid / sy) ** 2)
    dof = len(x) - len(popt)
    print(f"a = {popt[0]:.4g} ± {perr[0]:.2g}")
    print(f"b = {popt[1]:.4g} ± {perr[1]:.2g}")
    print(f"chi2/dof = {chi2 / dof:.2f}  (نزدیک ۱ = برازش و عدم‌قطعیت‌ها سازگارند)")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(6, 5), gridspec_kw={"height_ratios": [3, 1]}
    )
    xx = np.linspace(x.min(), x.max(), 200)
    ax1.errorbar(x, y, yerr=sy, fmt="o", capsize=3, label="data")
    ax1.plot(xx, model(xx, *popt), label="fit")
    ax1.set_ylabel("y [unit]")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.errorbar(x, resid, yerr=sy, fmt="o", capsize=3)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("x [unit]")
    ax2.set_ylabel("residual")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "fit.png", dpi=200)


if __name__ == "__main__":
    main(demo="--demo" in sys.argv)
