import logging
import os
import random
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.nvcf_enroot.NVCFEnrootProvider")
logger.setLevel(logging.INFO)

RETRY_INTERVAL = 5
WAIT_TIME = 3
DEFAULT_SIF_PATH = "/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/nvcf-osworld-eval/osworld-linux.sif"
DEFAULT_SQUASHFS_PATH = "/tmp/osworld-linux-eval-v2.sqsh"
API_PORT_RANGE = (15000, 19999)
VNC_PORT_RANGE = (18000, 22999)
CHROME_PORT_RANGE = (19000, 22999)
VLC_PORT_RANGE = (20000, 22999)
SQUASHFS_MAGIC = b"hsqs"


class PortAllocationError(Exception):
    pass


class NVCFEnrootProvider(Provider):
    """
    Enroot-backed provider for the prebuilt OSWorld SIF image.

    This is a local fallback for machines where Docker/Singularity are not
    installed but enroot is available. It extracts the squashfs payload from
    the SIF once, creates a per-run enroot rootfs, and exposes the same
    localhost ports as the singularity provider.
    """

    _port_allocation_lock = threading.Lock()

    def __init__(self, region: str = None):
        super().__init__(region)
        self.process: Optional[subprocess.Popen] = None
        self.process_pid: Optional[int] = None
        self.container_name: Optional[str] = None
        self._stdout_fh = None
        self._stderr_fh = None

        self.server_port = None
        self.chromium_port = None
        self.vnc_port = None
        self.vlc_port = None

        self._check_enroot_availability()

    @staticmethod
    def _check_enroot_availability():
        if shutil.which("enroot") is None:
            raise RuntimeError("enroot is not available. Install enroot or use another OSWorld provider.")

    @staticmethod
    def _check_port_available(port: int) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _find_available_port(self, min_port: int, max_port: int, max_attempts: int = 50) -> int:
        rng = random.SystemRandom()
        ports = list(range(min_port, max_port + 1))
        rng.shuffle(ports)

        for port in ports[:max_attempts]:
            if self._check_port_available(port):
                return port
        raise PortAllocationError(f"No available ports found in range {min_port}-{max_port}")

    def _allocate_ports(self) -> Tuple[int, int, int, int]:
        with NVCFEnrootProvider._port_allocation_lock:
            api_port = self._find_available_port(*API_PORT_RANGE)
            vnc_port = self._find_available_port(*VNC_PORT_RANGE)
            chrome_port = self._find_available_port(*CHROME_PORT_RANGE)
            vlc_port = self._find_available_port(*VLC_PORT_RANGE)
            return api_port, vnc_port, chrome_port, vlc_port

    @staticmethod
    def _find_squashfs_offset(sif_path: Path) -> int:
        overlap = len(SQUASHFS_MAGIC) - 1
        offset = 0
        previous = b""
        with sif_path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                data = previous + chunk
                idx = data.find(SQUASHFS_MAGIC)
                if idx >= 0:
                    return offset - len(previous) + idx
                previous = data[-overlap:]
                offset += len(chunk)
        raise RuntimeError(f"Could not find squashfs payload in SIF: {sif_path}")

    @classmethod
    def _ensure_squashfs(cls, sif_path: Path, squashfs_path: Path):
        if squashfs_path.exists() and squashfs_path.stat().st_size > 0:
            return

        squashfs_path.parent.mkdir(parents=True, exist_ok=True)
        offset = cls._find_squashfs_offset(sif_path)
        logger.info("Extracting squashfs payload from %s at byte offset %d", sif_path, offset)
        with sif_path.open("rb") as src, squashfs_path.open("wb") as dst:
            src.seek(offset)
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)

    @staticmethod
    def _enroot_env() -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("ENROOT_RUNTIME_PATH", f"/tmp/enroot-runtime-{os.environ.get('USER', 'user')}")
        env.setdefault("ENROOT_CACHE_PATH", f"/tmp/enroot-cache-{os.environ.get('USER', 'user')}")
        env.setdefault("ENROOT_DATA_PATH", f"/tmp/enroot-data-{os.environ.get('USER', 'user')}")
        for key in ("ENROOT_RUNTIME_PATH", "ENROOT_CACHE_PATH", "ENROOT_DATA_PATH"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        return env

    def _create_container(self, squashfs_path: Path, env: dict[str, str]):
        prefix = os.environ.get("NVCF_ENROOT_CONTAINER_NAME_PREFIX", "osworld-enroot")
        self.container_name = f"{prefix}-{os.getpid()}-{int(time.time())}-{random.randint(1000, 9999)}"
        cmd = ["enroot", "create", "-f", "-n", self.container_name, str(squashfs_path)]
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)

    def _wait_for_vm_ready(self, timeout: int = 300):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"http://localhost:{self.server_port}/screenshot",
                    timeout=(10, 10),
                )
                if response.status_code == 200:
                    return True
            except Exception:
                pass

            if self.process and self.process.poll() is not None:
                self._read_and_raise_error()

            logger.info("Checking if nvcf_enroot container is ready...")
            time.sleep(RETRY_INTERVAL)

        raise TimeoutError("nvcf_enroot failed to become ready within timeout period")

    def _read_and_raise_error(self):
        self._close_log_handles()
        log_dir = Path("/tmp/osworld_nvcf_enroot_logs")
        err_files = sorted(log_dir.glob("enroot_*.err"), reverse=True)
        out_files = sorted(log_dir.glob("enroot_*.out"), reverse=True)
        error_output = err_files[0].read_text(errors="replace") if err_files else ""
        stdout_output = out_files[0].read_text(errors="replace") if out_files else ""
        raise RuntimeError(
            f"enroot container exited unexpectedly. Return code: {self.process.returncode}\n"
            f"stdout:\n{stdout_output[-4000:]}\nstderr:\n{error_output[-4000:]}"
        )

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"):
        del path_to_vm, headless, os_type

        sif_path = Path(os.environ.get("NVCF_ENROOT_SIF_PATH", DEFAULT_SIF_PATH))
        squashfs_path = Path(os.environ.get("NVCF_ENROOT_SQUASHFS_PATH", DEFAULT_SQUASHFS_PATH))
        if not sif_path.exists():
            raise FileNotFoundError(f"OSWorld SIF image not found: {sif_path}")

        env = self._enroot_env()

        try:
            self.server_port, self.vnc_port, self.chromium_port, self.vlc_port = self._allocate_ports()
            self._ensure_squashfs(sif_path, squashfs_path)
            self._create_container(squashfs_path, env)

            log_dir = Path("/tmp/osworld_nvcf_enroot_logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            stdout_path = log_dir / f"enroot_{timestamp}.out"
            stderr_path = log_dir / f"enroot_{timestamp}.err"
            self._stdout_fh = open(stdout_path, "w")
            self._stderr_fh = open(stderr_path, "w")

            cmd = [
                "enroot",
                "start",
                "--root",
                "--rw",
                "-e",
                f"API_PORT={self.server_port}",
                "-e",
                f"VNC_PORT={self.vnc_port}",
                "-e",
                f"CHROME_PORT={self.chromium_port}",
                "-e",
                f"VLC_PORT={self.vlc_port}",
                self.container_name,
                "/usr/local/bin/entrypoint.sh",
            ]

            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=self._stdout_fh,
                stderr=self._stderr_fh,
                text=True,
                start_new_session=True,
            )
            self.process_pid = self.process.pid

            time.sleep(2)
            if self.process.poll() is not None:
                self._read_and_raise_error()

            logger.info(
                "Started nvcf_enroot container '%s' with PID %d "
                "(api=%d, vnc=%d, chrome=%d, vlc=%d)",
                self.container_name,
                self.process_pid,
                self.server_port,
                self.vnc_port,
                self.chromium_port,
                self.vlc_port,
            )
            self._wait_for_vm_ready(timeout=int(os.environ.get("NVCF_ENROOT_READY_TIMEOUT", "300")))
        except Exception:
            self.stop_emulator("")
            raise

    def _close_log_handles(self):
        for fh in (self._stdout_fh, self._stderr_fh):
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass
        self._stdout_fh = None
        self._stderr_fh = None

    def get_ip_address(self, path_to_vm: str) -> str:
        if not all([self.server_port, self.chromium_port, self.vnc_port, self.vlc_port]):
            raise RuntimeError("Container not started - ports not allocated")
        return f"localhost:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for nvcf_enroot provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        del path_to_vm, region, args, kwargs
        env = self._enroot_env()
        try:
            self._close_log_handles()
            if self.process_pid is not None:
                logger.info("Stopping enroot process group (PID: %d)", self.process_pid)
                try:
                    os.killpg(self.process_pid, signal.SIGTERM)
                    time.sleep(2)
                    if self.process and self.process.poll() is None:
                        os.killpg(self.process_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception as e:
                    logger.warning("Failed to kill enroot process group: %s", e)

            if self.container_name and os.environ.get("NVCF_ENROOT_KEEP_CONTAINER", "0") != "1":
                subprocess.run(
                    ["enroot", "remove", "-f", self.container_name],
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                )

            time.sleep(WAIT_TIME)
        except Exception as e:
            logger.error("Error stopping nvcf_enroot container: %s", e)
        finally:
            self.process = None
            self.process_pid = None
            self.container_name = None
            self.server_port = None
            self.chromium_port = None
            self.vnc_port = None
            self.vlc_port = None
