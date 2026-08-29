# Phase 0 failed iteration: missing `ensurepip`

- Status: repaired in a subsequent iteration
- UTC date: 2026-08-29
- Classification: external environment / bootstrap
- Command: `./scripts/bootstrap.sh`
- Repository commit before attempt: `efe860b154ccd7003a5660b17bdc35193694e153`
- Host Python: `Python 3.12.3`
- Seed: not applicable; failure occurred before any experiment
- Device/dtype: not applicable; failure occurred before PyTorch installation

## Frozen output

```text
The virtual environment was not created successfully because ensurepip is not
available.  On Debian/Ubuntu systems, you need to install the python3-venv
package using the following command.

    apt install python3.12-venv

You may need to use sudo with that command.  After installing the python3-venv
package, recreate your virtual environment.

Failing command: /root/aurelis/.venv/bin/python3
```

## Diagnosis and repair

Ubuntu splits `venv`/`ensurepip` into `python3.12-venv`. The package was absent;
APT reported candidate `3.12.3-1ubuntu0.16`. The repair is to install that
matching distribution package, remove only the incomplete repository-local
`.venv`, and rerun the non-destructive bootstrap. No ROCm, driver, kernel, or
system Python version is changed.

## Invariants and predicted effect

The repair only supplies Python's standard virtual-environment bootstrap. It
does not change AURELIS equations or numerical behavior. It should make
`python3 -m venv .venv` produce a local interpreter with pip. It would not
repair an incompatible Python wheel or a GPU runtime failure; those remain
separate gates.
