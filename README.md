# Nadir

SAR-conditioned cloud removal for Sentinel-2 imagery.

> **Status: work in progress (Phase 1).** This README is a stub and will be
> rewritten in full once experimental results are available.

## Quick start

```bash
# install (uv)
uv venv && uv pip install -e ".[dev]"
# or pip
pip install -e ".[dev]"

# generate dummy data and run the baseline training smoke test
python scripts/make_dummy_data.py
python -m nadir.train +experiment=dsen2cr_dummy
```
