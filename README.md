# 1972 Corvette C3 Suspension Redesign

Custom suspension development for a 1972 Corvette C3, converting the stock
SLA front and trailing-arm rear into a modern multi-link setup optimized for
HPDE track use while remaining street-legal.

## Project scope

- Vehicle dynamics simulation (Python, kinematics-first, later quasi-static)
- 3D measurement of existing chassis hardpoints
- Parametric design of front and rear suspension geometry
- CAD design and fabrication of custom components

## Structure

- `suspension/` — core Python modules (kinematic solver, hardpoint data, etc.)
- `notebooks/` — Jupyter notebooks for exploration and analysis
- `measurements/` — chassis measurement data
- `cad_exports/` — STEP/STL snapshots from Onshape (selective)
- `docs/` — design notes, decisions, references

## Status

In progress. See notebooks for current work.