import numpy as np, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from simulate import *

PHASE_COLOR = {"S1": "#1f77b4", "S2": "#ff7f0e", "S3": "#2ca02c", "COAST": "#999999"}
PHASE_LABEL = {
    "S1": "Stage 1 burn",
    "S2": "Stage 2 burn",
    "S3": "Stage 3 burn",
    "COAST": "Coast",
}


def gather(result):
    """Flatten all segments into continuous arrays plus phase boundaries."""
    T, ALT, SPD, DR, M, PH = [], [], [], [], [], []
    bounds = []
    for sol, name in zip(result["segments"], result["phase_names"]):
        tf = np.linspace(sol.t[0], sol.t[-1], 300)
        x, y, vx, vy, m = sol.sol(tf)
        T.append(tf)
        ALT.append(np.hypot(x, y) - R_EARTH)
        SPD.append(np.hypot(vx, vy))
        DR.append(np.arctan2(y, x) * R_EARTH)
        M.append(m)
        PH.append(name)
        bounds.append((tf[0], tf[-1], name))
    return T, ALT, SPD, DR, M, PH, bounds


def presentation_plot(result, target_alt=400e3, fname="present.png"):
    T, ALT, SPD, DR, M, PH, bounds = gather(result)
    tgt_v = np.sqrt(MU_EARTH / (R_EARTH + target_alt))
    cr = result["crossing"]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)
    fig.suptitle(
        "Total Orbital Flight Profile",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )

    # ---------- 1. Altitude vs time ----------
    ax = fig.add_subplot(gs[0, 0])
    for t, a, name in zip(T, ALT, PH):
        ax.plot(
            t,
            a / 1000,
            color=PHASE_COLOR[name],
            lw=2.5,
            label=(
                PHASE_LABEL[name]
                if PHASE_LABEL[name] not in [l.get_label() for l in ax.lines]
                else None
            ),
        )
    ax.axhline(100, color="k", ls="--", lw=1, alpha=0.5)
    ax.text(5, 108, "Kármán line — edge of space (100 km)", fontsize=8, alpha=0.7)
    ax.axhline(target_alt / 1000, color="crimson", ls=":", lw=1.5)
    ax.text(
        5, target_alt / 1000 + 18, "Target orbit (400 km)", fontsize=8, color="crimson"
    )
    for t0, t1, name in bounds:
        if name != "COAST":
            ax.axvline(t1, color="grey", lw=0.7, alpha=0.5)
    ax.plot(0, ALT[0][0] / 1000, "o", color="k", ms=8, zorder=5)
    ax.annotate(
        "Released from aircraft\n18 km, 240 m/s",
        (0, ALT[0][0] / 1000),
        textcoords="offset points",
        xytext=(28, -2),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Altitude", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)

    # ---------- 2. Speed vs time ----------
    ax = fig.add_subplot(gs[0, 1])
    for t, s, name in zip(T, SPD, PH):
        ax.plot(t, s / 1000, color=PHASE_COLOR[name], lw=2.5)
    ax.axhline(tgt_v / 1000, color="crimson", ls="--", lw=1.5)
    ax.text(
        5,
        tgt_v / 1000 + 0.16,
        f"Orbital speed ({tgt_v/1000:.2f} km/s)",
        fontsize=8.5,
        color="crimson",
        fontweight="bold",
    )
    if cr:
        ax.plot(cr["t"], cr["speed"] / 1000, "*", color="crimson", ms=20, zorder=5)
        ax.annotate(
            f"ORBITAL VELOCITY REACHED\nt = {cr['t']:.0f} s",
            (cr["t"], cr["speed"] / 1000),
            textcoords="offset points",
            xytext=(-165, -30),
            fontsize=9,
            fontweight="bold",
            color="crimson",
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2),
        )
    for t0, t1, name in bounds:
        if name != "COAST":
            ax.axvline(t1, color="grey", lw=0.7, alpha=0.5)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speed (km/s)")
    ax.set_title("Velocity", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.set_ylim(0, None)

    # ---------- 3. Whole-Earth view ----------
    ax = fig.add_subplot(gs[1, 0])
    circ = np.linspace(0, 2 * np.pi, 500)
    ax.fill(
        R_EARTH * np.cos(circ) / 1000,
        R_EARTH * np.sin(circ) / 1000,
        color="#a8c8e8",
        ec="#3b6ea5",
        lw=1.5,
        zorder=1,
    )
    ax.plot(
        (R_EARTH + 400e3) * np.cos(circ) / 1000,
        (R_EARTH + 400e3) * np.sin(circ) / 1000,
        color="crimson",
        ls=":",
        lw=1.2,
        zorder=2,
    )
    for d, a, name in zip(DR, ALT, PH):
        t_ = d / R_EARTH
        rr = (R_EARTH + a) / 1000
        ax.plot(
            rr * np.sin(t_), rr * np.cos(t_), color=PHASE_COLOR[name], lw=2.8, zorder=3
        )
    r0 = (R_EARTH + ALT[0][0]) / 1000
    ax.plot(0, r0, "o", color="k", ms=7, zorder=5)
    ax.annotate(
        "Release", (0, r0), textcoords="offset points", xytext=(-56, 6), fontsize=9
    )
    if cr:
        thc = None
        for d, a, t in zip(DR, ALT, T):
            idx = np.argmin(np.abs(t - cr["t"]))
            if abs(t[idx] - cr["t"]) < 5:
                thc = d[idx] / R_EARTH
                rc_ = (R_EARTH + a[idx]) / 1000
        if thc is not None:
            ax.plot(
                rc_ * np.sin(thc),
                rc_ * np.cos(thc),
                "*",
                color="crimson",
                ms=17,
                zorder=6,
            )
            ax.annotate(
                "Orbit reached",
                (rc_ * np.sin(thc), rc_ * np.cos(thc)),
                textcoords="offset points",
                xytext=(12, 10),
                fontsize=9,
                color="crimson",
                fontweight="bold",
            )
    ax.text(
        0,
        -R_EARTH / 2500,
        "EARTH",
        fontsize=13,
        ha="center",
        va="center",
        color="#22496e",
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        -(R_EARTH + 400e3) / 1000 * 0.70,
        (R_EARTH + 400e3) / 1000 * 0.70,
        "400 km target orbit",
        fontsize=8,
        color="crimson",
        rotation=45,
        ha="center",
    )
    ax.set_aspect("equal")
    lim = (R_EARTH + 1000e3) / 1000
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("Orbital Trajectory (to scale)", fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # ---------- 4. Mass ----------
    ax = fig.add_subplot(gs[1, 1])
    for t, m, name in zip(T, M, PH):
        ax.plot(t, m, color=PHASE_COLOR[name], lw=2.5)
    for t0, t1, name in bounds:
        if name != "COAST":
            ax.axvline(t1, color="grey", lw=0.7, alpha=0.5)
    ax.axhline(PAYLOAD_MASS, color="darkgreen", ls="--", lw=1.5)
    ax.text(
        5,
        PAYLOAD_MASS + 380,
        f"Satellite payload ({PAYLOAD_MASS:.0f} kg)",
        fontsize=8.5,
        color="darkgreen",
        fontweight="bold",
    )
    ax.annotate(
        "Stage 1 drops away",
        (bounds[0][1], M[0][-1]),
        textcoords="offset points",
        xytext=(30, 60),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Vehicle mass (kg)")
    ax.set_title("Mass", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)

    plt.savefig(fname, dpi=95, bbox_inches="tight")
    print("saved", fname)


rc = ReleaseConditions()
res = simulate(rc, [60, 20, 0], [0, 450, 0], verbose=False)
presentation_plot(res)
a, e, hp, ha = orbit_elements(res["final"])
print(f"final orbit {hp/1000:.0f} x {ha/1000:.0f} km e={e:.3f}")
print("segments:", res["phase_names"])
