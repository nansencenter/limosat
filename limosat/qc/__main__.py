"""Run LiMOSAT QC on one processing period (the v2 streaming QC remains ``python -m limosat.qc.archive``)."""

from .qc import main


if __name__ == "__main__":
    main()
