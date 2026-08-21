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


# ----------------------------------------------------------------------------
# Coast phase -- a Stage-shaped object with no propulsion, so propagate() and
# derivatives() need no special-casing. Duck typing, deliberately.
# ----------------------------------------------------------------------------
@dataclass
class Coast:
    duration: float
    name: str = "COAST"
    thrust: float = 0.0
    struct_mass: float = 0.0

    @property
    def mdot(self) -> float:
        return 0.0

    @property
    def burn_time(self) -> float:
        return self.duration


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


def orbital_velocity_event(t, state, stage, theta):
    """Zero-crossing when inertial speed reaches LOCAL circular velocity."""
    rmag = np.hypot(state[0], state[1])
    vmag = np.hypot(state[2], state[3])
    return vmag - np.sqrt(MU_EARTH / rmag)


orbital_velocity_event.terminal = False
orbital_velocity_event.direction = 1


def orbit_elements(state):
    """2D r,v -> (a, e, perigee_alt, apogee_alt). Negative a means hyperbolic."""
    x, y, vx, vy = state[:4]
    r = np.hypot(x, y)
    v2 = vx * vx + vy * vy
    energy = v2 / 2.0 - MU_EARTH / r
    h = x * vy - y * vx  # scalar ang. momentum in the plane
    a = -MU_EARTH / (2.0 * energy)
    e = np.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h / MU_EARTH**2))
    return a, e, a * (1 - e) - R_EARTH, a * (1 + e) - R_EARTH


def remaining_delta_v(mass, stage_index):
    """Ideal dV still available: rest of current stage + all stages above it."""
    if stage_index >= len(STAGES):
        return 0.0
    s = STAGES[stage_index]
    m_burnout = stack_mass(stage_index) - s.prop_mass
    dv = s.isp * G0 * np.log(mass / m_burnout) if mass > m_burnout else 0.0
    for j in range(stage_index + 1, len(STAGES)):
        sj = STAGES[j]
        m0 = stack_mass(j)
        dv += sj.isp * G0 * np.log(m0 / (m0 - sj.prop_mass))
    return dv


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
def build_sequence(coast_times_s):
    """Interleave stages with coast phases: [S1, coast, S2, coast, S3, coast]."""
    seq = []
    for i, stage in enumerate(STAGES):
        seq.append(stage)
        if coast_times_s[i] > 0.0:
            seq.append(Coast(duration=coast_times_s[i]))
    return seq


def simulate(rc, pitches_deg, coast_times_s=None, verbose=True):
    """
    pitches_deg   : constant pitch above LOCAL horizontal, one per powered stage.
    coast_times_s : coast duration AFTER each stage separates, one per stage.
                    Zeros give the old immediate-restaging behaviour.

    Returns dict with segments, thetas, final state, and the orbital-velocity
    crossing report (None if never reached).
    """
    if coast_times_s is None:
        coast_times_s = [0.0] * len(STAGES)

    state = initial_state(rc)
    t = 0.0
    segments, thetas, phase_names = [], [], []
    stage_idx = 0
    crossing = None

    for phase in build_sequence(coast_times_s):
        is_coast = isinstance(phase, Coast)
        # coast holds the last commanded attitude; powered phase re-references
        # its pitch to the local horizontal at ignition
        theta = (
            thetas[-1]
            if is_coast
            else pitch_to_theta(np.radians(pitches_deg[stage_idx]), state)
        )

        sol = solve_ivp(
            fun=derivatives,
            t_span=(t, t + phase.burn_time),
            y0=state,
            args=(phase, theta),
            method="DOP853",
            events=[impact, orbital_velocity_event],
            rtol=1e-10,
            atol=ATOL,
            dense_output=True,
        )
        segments.append(sol)
        thetas.append(theta)
        phase_names.append(phase.name)

        # first time inertial speed crosses local circular velocity
        if crossing is None and sol.t_events[1].size > 0:
            tc = sol.t_events[1][0]
            sc = sol.y_events[1][0]
            a, e, hp, ha = orbit_elements(sc)
            crossing = dict(
                t=tc,
                altitude=np.hypot(sc[0], sc[1]) - R_EARTH,
                speed=np.hypot(sc[2], sc[3]),
                mass=sc[4],
                prop_left=(
                    sc[4] - (stack_mass(stage_idx) - STAGES[stage_idx].prop_mass)
                    if stage_idx < len(STAGES)
                    else 0.0
                ),
                dv_left=remaining_delta_v(sc[4], stage_idx),
                perigee_alt=hp,
                apogee_alt=ha,
                eccentricity=e,
                phase=phase.name,
            )

        if sol.status == 1 and sol.t_events[0].size > 0:
            if verbose:
                print(f"  {phase.name}: IMPACT at t={sol.t[-1]:.1f}s")
            break

        t = sol.t[-1]
        state = sol.y[:, -1].copy()
        if not is_coast:
            state[4] -= phase.struct_mass  # jettison spent stage
            stage_idx += 1

        if verbose:
            alt = np.hypot(state[0], state[1]) - R_EARTH
            spd = np.hypot(state[2], state[3])
            tag = "coast end" if is_coast else "burnout  "
            print(
                f"  {phase.name:5s} {tag} | t={t:6.1f} s | alt={alt/1000:7.1f} km "
                f"| v={spd:7.1f} m/s | m={state[4]:7.1f} kg"
            )

    return dict(
        segments=segments,
        thetas=thetas,
        phase_names=phase_names,
        final=state,
        crossing=crossing,
    )


def report_orbit(state, crossing=None):
    a, e, hp, ha = orbit_elements(state)
    print("\nFinal orbit:")
    print(f"  perigee {hp/1000:8.1f} km   apogee {ha/1000:8.1f} km   e={e:.4f}")
    if hp < 100_000:
        print(f"  -> perigee inside the atmosphere: NOT a sustainable orbit")
    if crossing is None:
        print("\nNever reached local circular velocity.")
        return
    c = crossing
    print(f"\nOrbital velocity reached during {c['phase']} at t={c['t']:.1f} s:")
    print(f"  altitude       {c['altitude']/1000:8.1f} km")
    print(f"  speed          {c['speed']:8.1f} m/s")
    print(f"  mass           {c['mass']:8.1f} kg")
    print(f"  propellant left{c['prop_left']:8.1f} kg")
    print(f"  ideal dV left  {c['dv_left']:8.1f} m/s")
    print(
        f"  orbit at that instant: {c['perigee_alt']/1000:.1f} x {c['apogee_alt']/1000:.1f} km"
    )


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
PHASE_COLOR = {"S1": "tab:blue", "S2": "tab:orange", "S3": "tab:green", "COAST": "grey"}


def velocity_components(Y):
    """Split inertial velocity into LOCAL radial (climb) and tangential (orbital)."""
    x, y, vx, vy = Y[0], Y[1], Y[2], Y[3]
    rmag = np.hypot(x, y)
    v_radial = (x * vx + y * vy) / rmag
    v_tangential = (x * vy - y * vx) / rmag
    return v_radial, v_tangential


def plot_trajectory(result, target_alt=400_000.0):
    segments, names = result["segments"], result["phase_names"]
    v_orb = np.sqrt(MU_EARTH / (R_EARTH + target_alt))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    seen = set()

    for sol, name in zip(segments, names):
        tf = np.linspace(sol.t[0], sol.t[-1], 400)
        Y = sol.sol(tf)
        x, y, vx, vy, m = Y

        rmag = np.hypot(x, y)
        alt = rmag - R_EARTH
        speed = np.hypot(vx, vy)
        downrange = np.arctan2(y, x) * R_EARTH
        v_rad, v_tan = velocity_components(Y)

        c = PHASE_COLOR[name]
        lab = name if name not in seen else None
        seen.add(name)

        axes[0, 0].plot(downrange / 1e3, alt / 1e3, color=c, lw=2, label=lab)
        axes[0, 1].plot(tf, alt / 1e3, color=c, lw=2, label=lab)
        axes[1, 0].plot(tf, speed / 1e3, color=c, lw=2, label=lab)
        axes[1, 1].plot(tf, v_tan / 1e3, color=c, lw=2, label=lab)
        axes[1, 1].plot(tf, v_rad / 1e3, color=c, lw=1.4, ls="--", alpha=0.75)

    axes[0, 0].set(xlabel="Downrange (km)", ylabel="Altitude (km)", title="Trajectory")
    axes[0, 1].set(xlabel="Time (s)", ylabel="Altitude (km)", title="Altitude")
    axes[1, 0].set(xlabel="Time (s)", ylabel="Speed (km/s)", title="Inertial speed")
    axes[1, 1].set(
        xlabel="Time (s)",
        ylabel="Speed (km/s)",
        title="Velocity components: tangential (--) / radial (- -)",
    )

    axes[0, 1].axhline(target_alt / 1000, color="crimson", ls=":", lw=1.2)
    axes[1, 0].axhline(v_orb / 1000, color="crimson", ls="--", lw=1.2)
    axes[1, 1].axhline(v_orb / 1000, color="crimson", ls="--", lw=1.2)
    axes[1, 1].axhline(0.0, color="k", lw=0.8, ls=":")

    c = result.get("crossing")
    if c:
        for ax in (axes[1, 0], axes[1, 1]):
            ax.axvline(c["t"], color="crimson", lw=1.0, alpha=0.6)

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

    result = simulate(
        rc, pitches_deg=[60.0, 20.0, 5.0], coast_times_s=[0.0, 300.0, 0.0]
    )

    print("\nInsertion residuals [dr (m), dv (m/s), gamma (rad)]:")
    print(f"  {tgt.residuals(result['final'])}")

    report_orbit(result["final"], result["crossing"])

    plot_trajectory(result)
