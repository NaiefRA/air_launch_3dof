"""
3DOF air-launch-to-orbit trajectory simulation.

FRAME: 2D inertial, Earth-centered, planar.
    +x -> toward the release point at t=0  (so +x is local vertical ONLY at t=0)
    +y -> downrange / eastward
    State: [x, y, vx, vy, m]   (m, m, m/s, m/s, kg)

ANGLE CONVENTIONS -- read this before touching steering.
    theta  : thrust direction, measured CCW from inertial +x. This is what the
             integrator uses. It does NOT rotate with the vehicle's position.
    pitch  : thrust direction above the LOCAL horizontal at the current position.
             This is the human-facing number. Rotates with position.
    gamma  : VELOCITY direction above local horizontal (flight path angle).
    alpha  : angle between thrust axis and the AIRFLOW (v_rel). Aerodynamic AoA.

    Use pitch_to_theta() / theta_to_pitch() to convert. Never eyeball it.
"""

import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, minimize
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------------
MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6378137.0  # m
OMEGA_E = 7.2921159e-5  # rad/s
G0 = 9.80665  # m/s^2


# ----------------------------------------------------------------------------
# Release conditions
# ----------------------------------------------------------------------------
@dataclass
class ReleaseConditions:
    altitude: float = 18_000.0  # m    (sheet also says 16.5 km -- reconcile)
    carrier_speed: float = 240.0  # m/s  TAS at release
    carrier_fpa: float = 0.0  # deg  carrier climb angle at drop
    pitch_deg: float = 60.0  # deg  vehicle attitude above local horizontal at ignition
    ignition_delay: float = 0.0  # s    free-fall before S1 light (not yet modelled)

    @property
    def earth_rotation_speed(self) -> float:
        """Eastward inertial speed from Earth rotation.

        NOTE: this planar model is implicitly EQUATORIAL. A cos(latitude)
        factor here would be inconsistent with the omega x r used for the
        atmosphere in derivatives(). Keep one model, not two.
        """
        return OMEGA_E * (R_EARTH + self.altitude)


# ----------------------------------------------------------------------------
# Vehicle
# ----------------------------------------------------------------------------
@dataclass
class Stage:
    name: str
    isp: float  # s     (vacuum)
    thrust: float  # N     (vacuum; no back-pressure model -- see TODO)
    prop_mass: float  # kg
    struct_mass: float  # kg
    length: float  # m     bookkeeping only in 3DOF

    @property
    def mdot(self) -> float:
        return self.thrust / (self.isp * G0)

    @property
    def burn_time(self) -> float:
        return self.prop_mass / self.mdot

    @property
    def total_mass(self) -> float:
        return self.prop_mass + self.struct_mass


STAGES = [
    Stage(
        "S1",
        isp=320.0,
        thrust=157_850.0,
        prop_mass=6786.0,
        struct_mass=1014.0,
        length=8.50,
    ),
    Stage(
        "S2",
        isp=360.0,
        thrust=39_880.0,
        prop_mass=2754.0,
        struct_mass=306.0,
        length=5.20,
    ),
    Stage(
        "S3", isp=330.0, thrust=4_370.0, prop_mass=257.0, struct_mass=64.0, length=2.15
    ),
]

PAYLOAD_MASS = 315.0
FAIRING_MASS = 0.0  # TODO: NOT ON THE SPEC SHEET. Find this number.
FAIRING_JETTISON_ALT = 110_000.0

BODY_DIAMETER = 1.2
REF_AREA = np.pi * (BODY_DIAMETER / 2.0) ** 2
CD_CONST = 0.30  # TODO: replace with Cd(Mach) table from the OpenFOAM work

# TODO: S1 thrust is quoted without stating vacuum vs sea level, and no nozzle
# exit area is given, so there is no back-pressure model. ~1-2% error on S1.


# ----------------------------------------------------------------------------
# Target orbit
# ----------------------------------------------------------------------------
@dataclass
class TargetOrbit:
    altitude: float = 400_000.0

    @property
    def radius(self) -> float:
        return R_EARTH + self.altitude

    @property
    def velocity(self) -> float:
        return np.sqrt(MU_EARTH / self.radius)

    def residuals(self, state):
        """Insertion errors [radius, speed, fpa]. Feed these to fsolve."""
        r, v = state[:2], state[2:4]
        rmag, vmag = np.linalg.norm(r), np.linalg.norm(v)
        gamma = np.arcsin(np.dot(r, v) / (rmag * vmag))
        return np.array([rmag - self.radius, vmag - self.velocity, gamma])


# ----------------------------------------------------------------------------
# Mass bookkeeping
# ----------------------------------------------------------------------------
def stack_mass(from_stage: int) -> float:
    return PAYLOAD_MASS + FAIRING_MASS + sum(s.total_mass for s in STAGES[from_stage:])


LIFTOFF_MASS = stack_mass(0)


def ideal_delta_v():
    out = []
    for i, s in enumerate(STAGES):
        m0 = stack_mass(i)
        out.append(s.isp * G0 * np.log(m0 / (m0 - s.prop_mass)))
    return out


# ----------------------------------------------------------------------------
# Atmosphere -- US Standard 1976, valid to 86 km
# ----------------------------------------------------------------------------
_LAYERS = [
    (0, 288.15, -6.5e-3, 101325.0),
    (11000, 216.65, 0.0, 22632.1),
    (20000, 216.65, 1.0e-3, 5474.89),
    (32000, 228.65, 2.8e-3, 868.019),
    (47000, 270.65, 0.0, 110.906),
    (51000, 270.65, -2.8e-3, 66.9389),
    (71000, 214.65, -2.0e-3, 3.95642),
]


def atm_density(h):
    """h in metres -> kg/m^3. Zero above 86 km, clamped at 0 below sea level."""
    if h > 86_000.0:
        return 0.0
    if h < 0.0:
        h = 0.0
    R, g0 = 287.053, 9.80665
    h0, T0, L, P0 = next(l for l in reversed(_LAYERS) if h >= l[0])
    T = T0 + L * (h - h0)
    P = (
        P0 * (T / T0) ** (-g0 / (L * R))
        if L != 0
        else P0 * np.exp(-g0 * (h - h0) / (R * T0))
    )
    return P / (R * T)


# ----------------------------------------------------------------------------
# Angle helpers -- the only sanctioned way to convert between conventions
# ----------------------------------------------------------------------------
def local_vertical(state):
    """Inertial angle of the local 'up' direction at the current position."""
    return np.arctan2(state[1], state[0])


def pitch_to_theta(pitch_rad, state):
    """Pitch above LOCAL horizontal -> inertial thrust angle theta."""
    return local_vertical(state) + (np.pi / 2.0 - pitch_rad)


def theta_to_pitch(theta, state):
    """Inertial thrust angle theta -> pitch above LOCAL horizontal."""
    return np.pi / 2.0 - (theta - local_vertical(state))


def relative_velocity(state):
    """Velocity relative to the co-rotating atmosphere. Airspeed, not inertial."""
    x, y, vx, vy, _ = state
    v_atm = np.array([-OMEGA_E * y, OMEGA_E * x])  # omega x r in the plane
    return np.array([vx, vy]) - v_atm


def flight_angles(state, theta):
    """Diagnostics: (q, alpha, gamma). alpha is against AIRFLOW, gamma against inertial velocity."""
    x, y, vx, vy, _ = state
    r = np.array([x, y])
    v = np.array([vx, vy])
    rmag = np.linalg.norm(r)

    v_rel = relative_velocity(state)
    q = 0.5 * atm_density(rmag - R_EARTH) * np.dot(v_rel, v_rel)
    alpha = theta - np.arctan2(v_rel[1], v_rel[0])
    gamma = np.arcsin(np.dot(r, v) / (rmag * np.linalg.norm(v)))
    return q, alpha, gamma


# ----------------------------------------------------------------------------
# Dynamics
# ----------------------------------------------------------------------------
def derivatives(t, state, stage: Stage, theta: float):
    x, y, vx, vy, m = state
    r = np.array([x, y])
    r_mag = np.linalg.norm(r)

    # airspeed -- the atmosphere co-rotates with Earth, so aero uses v_rel not v
    v_rel = relative_velocity(state)
    v_rel_mag = np.linalg.norm(v_rel)

    q = 0.5 * atm_density(r_mag - R_EARTH) * v_rel_mag**2

    a_gravity = -MU_EARTH * r / r_mag**3
    a_thrust = (stage.thrust / m) * np.array([np.cos(theta), np.sin(theta)])
    a_drag = -(q * REF_AREA * CD_CONST / m) * (v_rel / v_rel_mag)

    a_net = a_gravity + a_thrust + a_drag
    return [vx, vy, a_net[0], a_net[1], -stage.mdot]


def impact(t, state, stage, theta):
    return np.linalg.norm(state[:2]) - R_EARTH


impact.terminal = True
impact.direction = -1

ATOL = np.array([1e-3, 1e-3, 1e-6, 1e-6, 1e-3])  # per-state: [m, m, m/s, m/s, kg]


# ----------------------------------------------------------------------------
# Initial state
# ----------------------------------------------------------------------------
def initial_state(rc: ReleaseConditions) -> np.ndarray:
    """
    Inertial state at release. Velocity is the CARRIER's velocity -- the
    pitch_deg on the spec sheet is an ATTITUDE, not a flight path angle, so
    the vehicle starts at ~pitch_deg of angle of attack.
    """
    r = R_EARTH + rc.altitude
    fpa = np.radians(rc.carrier_fpa)
    v_air = rc.carrier_speed * np.array([np.sin(fpa), np.cos(fpa)])
    v_rot = np.array([0.0, rc.earth_rotation_speed])
    v = v_air + v_rot
    return np.array([r, 0.0, v[0], v[1], LIFTOFF_MASS])


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def simulate(rc: ReleaseConditions, pitches_deg: list[float], verbose: bool = True):
    """
    pitches_deg: one constant pitch (above LOCAL horizontal at ignition) per stage.
    Converted to a fixed inertial theta held for that stage's whole burn.

    Returns (segments, final_state, thetas).
    """
    state = initial_state(rc)
    t = 0.0
    segments, thetas = [], []

    for stage, pitch_deg in zip(STAGES, pitches_deg):
        theta = pitch_to_theta(np.radians(pitch_deg), state)
        thetas.append(theta)

        sol = solve_ivp(
            fun=derivatives,
            t_span=(t, t + stage.burn_time),
            y0=state,
            args=(stage, theta),
            method="DOP853",
            events=[impact],
            rtol=1e-10,
            atol=ATOL,
            dense_output=True,
        )
        segments.append(sol)

        if sol.status == 1:
            if verbose:
                print(
                    f"  {stage.name}: terminal event fired at t={sol.t[-1]:.1f}s -- impact"
                )
            break

        t = sol.t[-1]
        state = sol.y[:, -1].copy()  # copy: sol.y[:, -1] is a view
        state[4] -= stage.struct_mass  # jettison spent stage

        if verbose:
            alt = np.linalg.norm(state[:2]) - R_EARTH
            spd = np.linalg.norm(state[2:4])
            print(
                f"  {stage.name} burnout | t={t:6.1f} s | alt={alt/1000:7.1f} km "
                f"| v={spd:7.1f} m/s | m={state[4]:7.1f} kg"
            )

    return segments, state, thetas


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_trajectory(segments, thetas):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for i, (sol, color) in enumerate(zip(segments, colors)):
        tf = np.linspace(sol.t[0], sol.t[-1], 400)
        Y = sol.sol(tf)
        x, y, vx, vy, m = Y

        rmag = np.hypot(x, y)
        alt = rmag - R_EARTH
        speed = np.hypot(vx, vy)
        downrange = np.arctan2(y, x) * R_EARTH

        q, alpha, gamma, pitch = [], [], [], []
        for k in range(len(tf)):
            s = Y[:, k]
            qk, ak, gk = flight_angles(s, thetas[i])
            q.append(qk)
            alpha.append(np.degrees(ak))
            gamma.append(np.degrees(gk))
            pitch.append(np.degrees(theta_to_pitch(thetas[i], s)))
        q = np.array(q)
        alpha = np.array(alpha)

        label = STAGES[i].name
        axes[0, 0].plot(downrange / 1e3, alt / 1e3, color=color, label=label)
        axes[0, 1].plot(tf, alt / 1e3, color=color, label=label)
        axes[0, 2].plot(tf, speed / 1e3, color=color, label=label)
        axes[1, 0].plot(tf, gamma, color=color, label=f"{label} gamma")
        axes[1, 0].plot(tf, pitch, color=color, ls="--", alpha=0.5)
        axes[1, 1].plot(tf, q / 1e3, color=color, label=f"{label} q")
        axes[1, 1].plot(tf, np.abs(alpha), color=color, ls=":", alpha=0.7)
        axes[1, 2].plot(tf, q * np.abs(np.radians(alpha)), color=color, label=label)

    axes[0, 0].set(xlabel="Downrange (km)", ylabel="Altitude (km)", title="Trajectory")
    axes[0, 1].set(xlabel="Time (s)", ylabel="Altitude (km)", title="Altitude")
    axes[0, 2].set(xlabel="Time (s)", ylabel="Speed (km/s)", title="Inertial speed")
    axes[1, 0].set(
        xlabel="Time (s)", ylabel="deg", title="gamma (--) / cmd pitch (- -)"
    )
    axes[1, 1].set(
        xlabel="Time (s)", ylabel="q (kPa) / alpha (deg)", title="Dyn. pressure & AoA"
    )
    axes[1, 2].set(
        xlabel="Time (s)", ylabel="q*alpha (Pa.rad)", title="Bending load proxy"
    )

    axes[1, 0].axhline(0, color="k", lw=0.8, ls=":")
    axes[1, 2].axhline(4000, color="r", lw=0.8, ls="--")  # placeholder structural limit

    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    rc = ReleaseConditions()
    tgt = TargetOrbit()

    print(f"Liftoff mass      : {LIFTOFF_MASS:.0f} kg")
    print(f"Earth rot. assist : {rc.earth_rotation_speed:.0f} m/s")
    print(f"Target v_circ     : {tgt.velocity:.0f} m/s\n")

    for s, dv in zip(STAGES, ideal_delta_v()):
        print(
            f"{s.name}: mdot={s.mdot:6.2f} kg/s  t_b={s.burn_time:6.1f} s  dV={dv:7.1f} m/s"
        )
    print(f"Total ideal dV    : {sum(ideal_delta_v()):.0f} m/s\n")

    segments, final, thetas = simulate(rc, pitches_deg=[60.0, 20.0, 5.0])

    print("\nInsertion residuals [dr (m), dv (m/s), gamma (rad)]:")
    print(f"  {tgt.residuals(final)}")

    plot_trajectory(segments, thetas)
