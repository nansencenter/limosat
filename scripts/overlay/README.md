# Chain container overlay (Olivia)

The chain runs in two layers:
- NVIDIA's PyTorch container, `/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif`. It is maintained by Sigma2.
- This overlay, mounted read-only on top of it. It holds the packages listed in `requirements.txt`, installed under `/home/apptainer/user_software`, which the chain finds through `PYTHONUSERBASE`.

## Build

The container is arm64, so build on a GPU node:

    sbatch --output=overlay-build.log scripts/overlay/build_overlay_olivia.sbatch \
        /cluster/projects/nn9878k/seachu/envs/efficientloftr-apptainer-v4.img "$PWD"

The job writes:
- the overlay image;
- `*.base-freeze.txt`: the base image's packages, used as pip constraints, so torch, numpy and the rest of the base image are never replaced;
- `*.lock.txt`: the packages installed into the overlay, with versions.

It then loads limosat and the EfficientLoFTR model on the GPU as a smoke test.

After the build:
- Commit the lock file here as `lock-<version>.txt`.
- Keep the overlay on `/cluster/projects` or NIRD, not `/cluster/work`. The work area deletes files whose modification times look old, and the v3 overlay was lost that way on 22 November 2026.

`submit_chain_olivia.sh` uses the v4 path by default. Set `OVERLAY=` to use another overlay.

## Pair-store identity

The matching fingerprint (`pair_identity.py`) covers the EfficientLoFTR source code and checkpoint, but not package versions. A new overlay must therefore reproduce stored pairs before it is used with an existing store. Rematch a sample of stored pairs into a scratch store and compare their content hashes.
