"""Friction-circle speed-profile solver and channel derivation.

This is an approximation of a rally car, not DiRT Rally 2.0's real physics
(which is proprietary). Calibrate `mu` and `power_w` against a recorded lap
with calibrate.py so the profile matches the game.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomalyDetection.generation.track import Track
from anomalyDetection.schema import DIST, TELEMETRY_COLUMNS

G = 9.81
V_TOP = 60.0

# Rough starting points; refine via calibration.
SURFACE_MU = {"tarmac": 1.05, "gravel": 0.65, "snow": 0.35, "ice": 0.20}


@dataclass
class CarParams:
    mass: float = 1250.0
    power_w: float = 250e3
    drivetrain_eff: float = 0.88
    drag_cda: float = 0.80
    air_density: float = 1.2
    rolling_resistance: float = 0.02
    wheelbase: float = 2.55
    track_width: float = 1.55
    wheel_radius: float = 0.33
    max_steer_angle: float = 0.45  # rad at the road wheel
    gear_ratios: tuple = (3.2, 2.3, 1.75, 1.4, 1.15, 0.98)
    final_drive: float = 4.4
    idle_rpm: float = 1500.0
    shift_rpm: float = 7000.0
    max_rpm: float = 7800.0


def power_accel(v, car: CarParams):
    return car.power_w * car.drivetrain_eff / (car.mass * np.maximum(v, 3.0))


def resist_decel(v, car: CarParams):
    return 0.5 * car.air_density * car.drag_cda * v**2 / car.mass + car.rolling_resistance * G


def grip_long(v, kappa, mu):
    """Longitudinal accel left over after cornering (friction circle)."""
    a_lat = v * v * abs(kappa)
    return np.sqrt(max((mu * G) ** 2 - a_lat**2, 0.0))


def brake_decel(v, kappa, car: CarParams, mu: float, frac: float = 1.0):
    return frac * grip_long(v, kappa, mu) + resist_decel(v, car)


def speed_profile(track: Track, car: CarParams, mu: float, v_start: float = 0.0) -> np.ndarray:
    """Time-optimal speed along the stage: min(forward accel pass, backward brake pass)."""
    k, ds, n = np.abs(track.kappa), track.ds, len(track.s)
    v_lat = np.minimum(np.sqrt(mu * G / np.maximum(k, 1e-6)), V_TOP)

    fwd = v_lat.copy()
    fwd[0] = min(v_start, v_lat[0])
    for i in range(n - 1):
        v = max(fwd[i], 3.0)
        a = min(power_accel(v, car) - resist_decel(v, car), grip_long(v, k[i], mu))
        fwd[i + 1] = min(v_lat[i + 1], np.sqrt(max(fwd[i] ** 2 + 2 * a * ds, 0.0)))

    bwd = v_lat.copy()
    for i in range(n - 2, -1, -1):
        v = bwd[i + 1]
        bwd[i] = min(v_lat[i], np.sqrt(v * v + 2 * brake_decel(v, k[i + 1], car, mu) * ds))
    return np.minimum(fwd, bwd)


def gentle_braking_from(v: np.ndarray, track: Track, car: CarParams, mu: float,
                        apex: int, brake_start: int, frac: float) -> tuple[np.ndarray, int]:
    """Re-brake into `apex` using only `frac` of available braking.

    Returns (new profile, first modified index). Models a driver braking early.
    `brake_start` is where the original braking began; the search cannot stop
    before it, so profile noise can't end it prematurely.
    """
    k, ds = np.abs(track.kappa), track.ds
    out = v.copy()
    j = apex - 1
    while j >= 0:
        vn = out[j + 1]
        e = np.sqrt(vn * vn + 2 * brake_decel(vn, k[j + 1], car, mu, frac) * ds)
        if j < brake_start and e > v[j] + 1e-6:  # rejoined the original profile
            break
        out[j] = min(e, v[j])
        j -= 1
    changed = np.flatnonzero(out < v - 1e-6)
    return out, int(changed[0]) if len(changed) else apex


def rpm_for(v, gear, car: CarParams):
    ratio = np.asarray(car.gear_ratios)[np.asarray(gear, dtype=int) - 1]
    rpm = v / car.wheel_radius * ratio * car.final_drive * 60 / (2 * np.pi)
    return np.clip(rpm, car.idle_rpm, car.max_rpm)


def gear_for_speed(v: np.ndarray, car: CarParams) -> np.ndarray:
    ratios = np.asarray(car.gear_ratios)
    rpm = v[:, None] / car.wheel_radius * ratios[None, :] * car.final_drive * 60 / (2 * np.pi)
    ok = rpm <= car.shift_rpm
    return np.where(ok.any(axis=1), ok.argmax(axis=1) + 1, len(ratios))


def derive_channels(track: Track, v: np.ndarray, car: CarParams, mu: float) -> pd.DataFrame:
    """Turn a speed profile into the full telemetry channel set."""
    ds = track.ds
    dt = ds / np.maximum(0.5 * (v[:-1] + v[1:]), 0.5)
    time = np.r_[0.0, np.cumsum(dt)]

    a_long = np.gradient(0.5 * v * v, ds)
    demand = a_long + resist_decel(v, car)  # accel the drivetrain / brakes must supply
    capacity = np.minimum(power_accel(v, car), mu * G)
    throttle = np.clip(demand / capacity, 0, 1)
    brake = np.clip(-demand / (mu * G), 0, 1)
    brake[brake < 0.03] = 0.0

    steer = np.clip(np.arctan(car.wheelbase * track.kappa) / car.max_steer_angle, -1, 1)
    gear = gear_for_speed(v, car)
    half = 0.5 * car.track_width * track.kappa  # left turn: right wheels are outer
    return pd.DataFrame({
        DIST: track.s, "time": time, "x": track.x, "z": track.z,
        "speed": v, "gear": gear, "rpm": rpm_for(v, gear, car),
        "throttle": throttle, "brake": brake, "steer": steer,
        "g_lat": v * v * track.kappa / G, "g_long": a_long / G,
        "wheel_speed_fl": v * (1 - half), "wheel_speed_fr": v * (1 + half),
        "wheel_speed_rl": v * (1 - half), "wheel_speed_rr": v * (1 + half),
    })[TELEMETRY_COLUMNS]
