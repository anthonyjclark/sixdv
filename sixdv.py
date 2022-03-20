from __future__ import annotations

from dataclasses import dataclass
from math import inf

import numpy as np
from numpy import cos, sin


@dataclass
class Body:
    # Index of parent body (None if root)
    parent: Body | None
    # Plücker coordinate transforms
    xtree: np.ndarray  # TODO: Pluker
    # Pitch/type of each joint (revolute, prismatic, helical)
    pitch: float
    # Inertia expressed in link coordinates
    inertia: np.ndarray  # TODO: SpatialInertia

    s: np.ndarray
    Xup: np.ndarray
    v: np.ndarray
    a: np.ndarray
    f: np.ndarray  # TODO: | None


class Model:
    """General kinematic tree (no cyles).

    A rigid-body system in which the connectivity is that of a topological tree.

    A sphereical joint can be emulated by a chain of three revolute joints.
    """

    def __init__(self, bodies: list[Body]):
        self.bodies = bodies


def rx(rad: float) -> np.ndarray:
    c, s = cos(rad), sin(rad)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def ry(rad: float) -> np.ndarray:
    c, s = cos(rad), sin(rad)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def rz(rad: float) -> np.ndarray:
    c, s = cos(rad), sin(rad)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def _rot(E: np.ndarray) -> np.ndarray:
    X = np.zeros((6, 6))
    X[:3, :3] = E
    X[-3:, -3:] = E
    return X


def rotx(rad: float) -> np.ndarray:
    return _rot(rx(rad))


def roty(rad: float) -> np.ndarray:
    return _rot(ry(rad))


def rotz(rad: float) -> np.ndarray:
    return _rot(rz(rad))


def _skew(x: float, y: float, z: float) -> np.ndarray:
    """Skew-symmetric matrix."""
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def xlt(r: np.ndarray):
    """r is R^3"""
    X = np.ones((6, 6))
    X[:3, -3:] = 0
    X[-3:, :3] = _skew(*r)
    return X


def crm(v: np.ndarray) -> np.ndarray:
    """v is R^6"""
    vskew = np.zeros((6, 6))
    vskew[:3, :3] = _skew(*v[:3])
    vskew[-3:, :3] = _skew(*v[3:])
    vskew[-3:, -3:] = _skew(*v[:3])
    return vskew


def crf(v: np.ndarray) -> np.ndarray:
    """Return dual of crm."""
    return -crm(v).T


def mcl(m: float, c: np.ndarray, Ic: np.ndarray) -> np.ndarray:
    """c is R^3 and Ic is R^3x3"""
    I = np.zeros((6, 6))
    cskew = _skew(*c)
    I[:3, :3] = Ic - m * cskew * cskew
    I[:3, -3:] = m * cskew
    I[-3:, :3] = -m * cskew
    I[-3:, -3:] = m
    return I


def XtoV(X: np.ndarray) -> np.ndarray:
    # TODO: very suspect...
    return 0.5 * np.array(
        [
            # Is the order correct?
            X[2, 3] - X[3, 2],
            X[3, 1] - X[1, 3],
            X[1, 2] - X[2, 1],
            X[5, 3] - X[6, 2],
            X[6, 1] - X[4, 3],
            X[4, 2] - X[5, 1],
        ]
    )


def jcalc(pitch: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    # Revolute joint
    if pitch == 0:
        XJ = rotz(q)
        s = np.array([0, 0, 1, 0, 0, 0])

    # Prismatic joint
    elif pitch == inf:
        XJ = xlt(np.array([0, 0, q]))
        s = np.array([0, 0, 0, 0, 0, 1])

    # Helical joint
    else:
        XJ = rotz(q) * xlt(np.array([0, 0, q * pitch]))
        s = np.array([0, 0, 1, 0, 0, pitch])

    return XJ, s


def inverse_dynamics(model: Model, q, qd, qdd):

    # TODO: make q, qd, qddd, and loop variables part of Body?

    tau = np.zeros((len(model.bodies), 1))

    for i, body in enumerate(model.bodies):

        XJ, body.s = jcalc(body.pitch, q[i])
        vJ = body.s * qd[i]
        body.Xup = XJ * body.xtree

        if body.parent is None:
            body.v = vJ
            body.a = body.Xup * np.array([0, 0, 0, 0, 0, 9.81]) + body.s * qdd[i]

        else:
            body.v = body.Xup * body.parent.v + vJ
            body.a = body.Xup * body.parent.a + body.s * qdd[i] + crm(body.v) * vJ

        body.f = body.inertia * body.a + crf(body.v) * body.inertia * body.v

    for i, body in reversed(list(enumerate(model.bodies))):

        tau[i, 1] = body.s.T * body.f

        if body.parent is not None:
            body.parent.f += body.Xup.T * body.f

    return tau
