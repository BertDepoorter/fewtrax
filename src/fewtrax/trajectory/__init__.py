"""Trajectory module: ODE-based orbital evolution for EMRI systems."""

from fewtrax.trajectory.inspiral import EMRIInspiral, run_inspiral

__all__ = ["EMRIInspiral", "run_inspiral"]
