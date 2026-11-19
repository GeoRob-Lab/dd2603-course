"""Files that come from rby1-sdk's repository, downloaded on first use.

The ``rby1-sdk`` pip package contains only the client library, not the robot models or
the protocol files, so these are fetched from rby1-sdk's git repository at the same
version as the pip package (v0.10.0), once, into ``.cache/`` at the repository root.
Only the folders needed are downloaded:

- ``models/*/mujoco``: Rainbow Robotics' MuJoCo models of the RB-Y1,
- ``models/*/urdf``: the URDF the robot reports to rby1-sdk clients,
- ``protos``: the robot's gRPC protocol.

Nothing here is stored in this repository.
"""

import subprocess
import sys
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache"

RBY1_SDK_URL = "https://github.com/RainbowRobotics/rby1-sdk.git"
RBY1_SDK_COMMIT = "9af8a734b7bef0167545e3d9f0d559276a5b64ee"  # tag v0.10.0, as the pip package
RBY1_SDK_FILES = ["/protos/", "/models/*/urdf/*.urdf", "/models/*/mujoco/"]  # sparse-checkout patterns
RBY1_SDK_DIR = CACHE_DIR / f"rby1-sdk-{RBY1_SDK_COMMIT[:7]}"

#: RB-Y1 variants: name -> (rby1-sdk model letter, MuJoCo model, URDF), paths in rby1-sdk.
MODELS = {
    f"rby1{base}_{version}": (base, f"models/rby1{base}/mujoco/model_v{version}.xml",
                              f"models/rby1{base}/urdf/model_v{version}.urdf")
    for base, versions in (("a", ("1.0", "1.1", "1.2")), ("m", ("1.0", "1.1", "1.2", "1.3")))
    for version in versions
}
DEFAULT_MODEL = "rby1a_1.0"  # the version of our robot


def rby1_sdk_dir() -> Path:
    """rby1-sdk's repository files (see the module docstring), downloaded on first use."""
    git = ["git", "-C", str(RBY1_SDK_DIR)]
    marker = RBY1_SDK_DIR / ".git" / "rbylearn-files"
    if not marker.is_file() or marker.read_text() != " ".join(RBY1_SDK_FILES):
        print(f"Downloading robot models and protocol files from {RBY1_SDK_URL} into {RBY1_SDK_DIR} ...", flush=True)
        RBY1_SDK_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run([*git, "init", "-q"], check=True)
        subprocess.run([*git, "remote", "add", "origin", RBY1_SDK_URL], check=False, stderr=subprocess.DEVNULL)
        subprocess.run([*git, "sparse-checkout", "set", "--no-cone", *RBY1_SDK_FILES], check=True)
        subprocess.run([*git, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", RBY1_SDK_COMMIT], check=True)
        subprocess.run([*git, "checkout", "-q", "FETCH_HEAD"], check=True)
        marker.write_text(" ".join(RBY1_SDK_FILES))
    return RBY1_SDK_DIR


def _check(model: str) -> None:
    if model not in MODELS:
        raise ValueError(f"Unknown RB-Y1 model '{model}'. Choose one of {list(MODELS)}.")


def rby1_mjcf_path(model: str = DEFAULT_MODEL) -> Path:
    """rby1-sdk's MuJoCo model file of an RB-Y1 variant."""
    _check(model)
    return rby1_sdk_dir() / MODELS[model][1]


def rby1_urdf(model: str = DEFAULT_MODEL) -> str:
    """rby1-sdk's URDF of an RB-Y1 variant, as the robot reports it to clients."""
    _check(model)
    return (rby1_sdk_dir() / MODELS[model][2]).read_text()


def import_rby1_api():
    """Import the Python gRPC code of the robot's API (``rb.api``), generating it on first use."""
    out = CACHE_DIR / "rb_api_python"
    if not (out / "rb" / "api").is_dir():
        from grpc_tools import protoc

        protos = rby1_sdk_dir() / "protos"
        out.mkdir(parents=True, exist_ok=True)
        files = [str(p.relative_to(protos)) for p in protos.rglob("*.proto")]
        args = ["protoc", f"-I{protos}", f"-I{Path(protoc.__file__).parent / '_proto'}",
                f"--python_out={out}", f"--grpc_python_out={out}", *files]
        if protoc.main(args) != 0:
            raise RuntimeError("Could not generate the Python code of the rby1-sdk protos.")
    if str(out) not in sys.path:
        sys.path.insert(0, str(out))
    import rb.api

    return rb.api


if __name__ == "__main__":  # python -m rbylearn.sim.models: download everything now
    import_rby1_api()
    print(f"rby1-sdk files are in {rby1_sdk_dir()}")
