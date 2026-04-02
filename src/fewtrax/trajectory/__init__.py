"""Trajectory module: ODE-based orbital evolution for EMRI systems."""

from fewtrax.trajectory.inspiral import EMRIInspiral, EMRIInspiralFast, run_inspiral

__all__ = ["EMRIInspiral", "EMRIInspiralFast", "run_inspiral"]
