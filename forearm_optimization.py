"""Nonlinear optimization of forearm (elbow flexor) muscle forces.

Solves for the muscle forces F = (F1, F2, F3) -- Brachialis, Biceps Brachii,
and Brachioradialis -- that minimize the physiological cost function

    u = ( sum_i ( F_i / PCSA_i ) ** n ) ** (1 / n)

subject to moment equilibrium about the elbow (sum M = 0) and the
physiological requirement that muscle forces are nonnegative (F_i >= 0).

The exponent n is a free parameter of the cost function: n=2 corresponds to
minimizing summed squared muscle stress, n=4 penalizes large individual
muscle stresses much more steeply, pushing the solution toward more evenly
shared (lower peak) stress across the three muscles.
"""

import numpy as np
from scipy.optimize import minimize


def _cost_func(F, pcsa, n):
    return np.sum((F / pcsa) ** n) ** (1.0 / n)


def solve_forearm_forces(
    n=2,
    F0=None,
    Fb=20.0,
    Fg=10.0,
    a=(2.5, 5.2, 18.7),
    c=30.0,
    b=15.0,
    theta=(80.0, 70.0, 20.0),
    pcsa=(8.4, 7.8, 4.7),
):
    """Solve for the muscle forces F=(F1,F2,F3) that minimize
    u = (sum((F/pcsa)**n))**(1/n), subject to moment equilibrium about the
    elbow and F >= 0.

    n: exponent of the physiological cost function
    F0: initial guess for (F1, F2, F3); defaults to (0, 0, 0)
    Fb, Fg: weight of book in hand (20N here), weight of forearm (bone)
    a: moment arms of the three muscles
    c: forearm length (radius-ulna)
    b: forearm center-of-mass distance
    theta: angle of each muscle's line of action, in degrees
    pcsa: physiological cross-sectional area (or max stress capacity) of each
        muscle, used to normalize the cost function

    Returns the scipy.optimize.OptimizeResult from the solve; result.x holds
    the optimal (F1, F2, F3).
    """

    a1, a2, a3 = a
    theta1, theta2, theta3 = np.radians(theta)
    pcsa = np.asarray(pcsa, dtype=float)

    def moment_equilibrium(F):
        return (
            a1 * np.sin(theta1) * F[0]
            + a2 * np.sin(theta2) * F[1]
            + a3 * np.sin(theta3) * F[2]
            - (Fb * c + Fg * b)
        )

    constraints = [{"type": "eq", "fun": moment_equilibrium}]
    bounds = [(0.0, None)] * 3
    if F0 is None:
        F0 = np.zeros(3)
    else:
        F0 = np.asarray(F0, dtype=float)

    return minimize(
        _cost_func,
        F0,
        args=(pcsa, n),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )


def solve_jrf(F1, F2, F3, theta=(80.0, 70.0, 20.0), Fb=20.0, Fg=10.0):
    """Compute the joint reaction force (JRF) at the elbow given the muscle
    forces and their line-of-action angles.

    Returns (JointX, JointY, magnitude, angle_degrees).
    """
    theta1, theta2, theta3 = np.radians(theta)

    JointX = -F1 * np.cos(theta1) - F2 * np.cos(theta2) - F3 * np.cos(theta3)
    JointY = F1 * np.sin(theta1) + F2 * np.sin(theta2) + F3 * np.sin(theta3) - Fb - Fg

    mag = np.sqrt(JointX ** 2 + JointY ** 2)
    angle = np.arctan(JointY / JointX)
    angled = 180 * angle / np.pi

    return JointX, JointY, mag, angled
