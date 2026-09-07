import json
import multiprocessing
import subprocess
import time
import os
import random
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional
import tyro
import wandb
import shlex
from utils import get_subset_json_files, load_remote_credentials
from typing import Dict, Literal
import shutil

# ===== LOGGING SETUP (DO THIS FIRST) =====
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"render_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(process)d | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Full log: {LOG_FILE}")

OBJAVERSE_INFO = 'info/objaverse'
SAVE_ROOT = os.path.abspath('rendered')

@dataclass
class Args:
    workers_per_gpu: int = 16
    """number of workers per gpu"""

    split: str = 'training'

    blender: str = ''
    """Blender executable path"""

    log_to_wandb: bool = True
    """Whether to log progress to wandb"""

    gpus: list = field(default_factory=lambda: [0,1,2,3,4,5,6,7,8,9])
    """GPU indices to use"""

    skip_exist: bool = True
    """Skip existing renders (checks for finish.keep marker)"""

    seed: int = 0
    """Random seed"""

    upload: bool = False
    """Upload rendered subsets to remote server after completion"""

    download: bool = False
    """If True, sync rendered data from remote BEFORE rendering. 
       Deletes info.json to allow re-rendering of already-completed objects."""

    upload_retries: int = 3
    """Number of retry attempts for failed uploads"""

    remote_ip: Optional[str] = None
    """Remote server IP (loaded from env/credentials file if not provided)"""

    remote_user: Optional[str] = None
    """Remote server IP (loaded from env/credentials file if not provided)"""

    remote_pwd: Optional[str] = None
    """Remote server password (loaded from env/credentials file if not provided)"""

    remote_path: Optional[str] = None
    """Remote base path (loaded from env/credentials file if not provided)"""

    credentials_file: Optional[str] = None
    """Path to credentials JSON file (default: ./credentials.json)."""

def upload_and_cleanup(
    local_path: str,
    remote_ip: str,
    remote_user: str,
    remote_auth: Dict[str, str],  # {'method': 'password', 'value': 'pwd'} OR {'method': 'key', 'value': '/path/to/key'}
    remote_path: str,
    max_retries: int = 3
) -> bool:
    """
    Upload directory to remote server with retry logic and local cleanup.
    Supports BOTH password AND SSH key authentication securely.
    
    Args:
        local_path: Local directory path to upload
        remote_ip: Remote server IP/hostname
        remote_user: Remote username
        remote_auth: Dict with keys:
            - method: 'password' or 'key'
            - value: password string OR path to private key file
        remote_path: Base remote path (subset hierarchy preserved under this)
        max_retries: Maximum upload attempts (default: 3)
    
    Returns:
        True on success, False if all retries fail.
    
    Security Notes:
        - Password auth requires sshpass and uses SSHPASS env var (safer than command-line args)
        - SSH keys should have permissions 600 (owner read/write only)
        - Local files are ONLY deleted after successful remote transfer verification
    """
    # Validate local path exists
    if not os.path.exists(local_path):
        logger.warning(f"Upload skipped - path doesn't exist: {local_path}")
        return False
    
    # Validate auth method
    auth_method: Literal['password', 'key'] = remote_auth.get('method')  # type: ignore
    auth_value = remote_auth.get('value', '')
    
    if auth_method not in ('password', 'key'):
        raise ValueError(f"Invalid auth method: {auth_method}. Must be 'password' or 'key'")
    
    if auth_method == 'key':
        if not os.path.exists(auth_value):
            raise FileNotFoundError(f"SSH key not found: {auth_value}")
        # Security: warn if key permissions are too permissive
        key_stat = os.stat(auth_value)
        if key_stat.st_mode & 0o077:
            logger.warning(
                f"SSH key permissions too open: {auth_value} (mode={oct(key_stat.st_mode & 0o777)}). "
                "Recommended: chmod 600 {auth_value}"
            )
    else:  # password auth
        if not shutil.which('sshpass'):
            raise RuntimeError("sshpass not found. Install it or use SSH key authentication.")
    
    # Compute relative path under SAVE_ROOT for remote hierarchy preservation
    try:
        subset_relpath = os.path.relpath(local_path, SAVE_ROOT)
        if subset_relpath.startswith('..') or subset_relpath == '.':
            raise ValueError(f"local_path '{local_path}' is not under SAVE_ROOT '{SAVE_ROOT}'")
    except Exception as e:
        logger.error(f"Failed to compute relative path: {e}")
        subset_relpath = os.path.basename(local_path)
    
    remote_full_path = os.path.join(remote_path, subset_relpath)
    logger.info(f"Upload target: {subset_relpath} -> {remote_user}@{remote_ip}:{remote_full_path}")
    
    # Retry loop
    for attempt in range(1, max_retries + 1):
        logger.info(f"Upload attempt {attempt}/{max_retries} for {subset_relpath}")
        error_msg = ""
        
        try:
            # === STEP 1: Create remote directory ===
            mkdir_cmd = ['ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=10']
            env = None
            
            if auth_method == 'password':
                mkdir_cmd = ['sshpass', '-e'] + mkdir_cmd
                env = os.environ.copy()
                env['SSHPASS'] = auth_value
            else:  # key auth
                mkdir_cmd += ['-i', auth_value]
            
            mkdir_cmd += [
                f'{remote_user}@{remote_ip}',
                f'mkdir -p {shlex.quote(remote_full_path)}'
            ]
            
            subprocess.run(
                mkdir_cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=True
            )
            logger.debug(f"Remote directory created: {remote_full_path}")
            
            # === STEP 2: Rsync transfer ===
            rsync_cmd = ['rsync', '-avz', '--exclude=finish.keep']
            env = None
            
            if auth_method == 'password':
                rsync_cmd = ['sshpass', '-e'] + rsync_cmd
                rsync_cmd += [
                    '-e', 'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10',
                    f'{local_path.rstrip("/")}/',
                    f'{remote_user}@{remote_ip}:{remote_full_path.rstrip("/")}/'
                ]
                env = os.environ.copy()
                env['SSHPASS'] = auth_value
            else:  # key auth
                rsync_cmd += [
                    '-e', f'ssh -i {shlex.quote(auth_value)} -o StrictHostKeyChecking=no -o ConnectTimeout=10',
                    f'{local_path.rstrip("/")}/',
                    f'{remote_user}@{remote_ip}:{remote_full_path.rstrip("/")}/'
                ]
            
            # Stream rsync output with minimal noise
            process = subprocess.Popen(
                rsync_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            noise_patterns = {
                'sending incremental file list', 'building file list',
                'total:', 'speedup is', 'bytes/sec'
            }
            
            for line in process.stdout or []:
                stripped = line.strip()
                if stripped and not any(pat in stripped for pat in noise_patterns):
                    logger.debug(f"RSYNC: {stripped}")
            
            ret = process.wait(timeout=3600)
            if ret != 0:
                raise RuntimeError(f"RSYNC failed with exit code {ret}")
            logger.debug("RSYNC transfer completed successfully")
            
            # === STEP 3: Verify remote files exist before cleanup ===
            verify_cmd = ['ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=10']
            env = None
            
            if auth_method == 'password':
                verify_cmd = ['sshpass', '-e'] + verify_cmd
                env = os.environ.copy()
                env['SSHPASS'] = auth_value
            else:
                verify_cmd += ['-i', auth_value]
            
            verify_cmd += [
                f'{remote_user}@{remote_ip}',
                f'test -d {shlex.quote(remote_full_path)} && find {shlex.quote(remote_full_path)} -type f 2>/dev/null | head -1'
            ]
            
            result = subprocess.run(
                verify_cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=15
            )
            if result.returncode != 0 or not result.stdout.strip():
                raise RuntimeError("Remote verification failed: no files found on destination")
            logger.debug("Remote verification succeeded")
            
            # === STEP 4: Cleanup local files (only after successful verification) ===
            removed_count = 0
            for root, _, files in os.walk(local_path):
                for file in files:
                    if file == "finish.keep":
                        continue
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")
            
            # Create finish.keep marker
            marker_path = os.path.join(local_path, "finish.keep")
            with open(marker_path, 'w') as f:
                f.write(f"Upload completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source: {local_path}\n")
                f.write(f"Destination: {remote_user}@{remote_ip}:{remote_full_path}\n")
                f.write(f"Auth method: {auth_method}\n")
                f.write(f"Files removed: {removed_count}\n")
                f.write(f"Upload attempts: {attempt}/{max_retries}\n")
            
            logger.info(
                f"✓ Upload & cleanup succeeded for {subset_relpath} on attempt {attempt} "
                f"({removed_count} files removed, marker: {marker_path})"
            )
            return True
            
        except subprocess.TimeoutExpired as e:
            error_msg = f"Timeout after {e.timeout}s during {e.cmd[0] if hasattr(e, 'cmd') else 'operation'}"
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed (exit {e.returncode}): {e.stderr.strip() if e.stderr else e.stdout.strip() if e.stdout else 'no output'}"
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
        
        logger.warning(f"Upload attempt {attempt} failed for {subset_relpath}: {error_msg}")
        
        if attempt < max_retries:
            backoff = min(2 ** (attempt - 1), 30)  # Exponential backoff capped at 30s
            logger.info(f"Retrying in {backoff}s...")
            time.sleep(backoff)
        else:
            logger.error(
                f"✗ All {max_retries} upload attempts failed for {subset_relpath}. "
                "Local files preserved (no cleanup performed)."
            )
            return False
    
    return False  # Should never reach here

def download_from_remote(
    local_path: str,
    remote_ip: str,
    remote_user: str,
    remote_auth: Dict[str, str],
    remote_path: str,
    max_retries: int = 3
) -> bool:
    """
    Download rendered subset from remote server to local machine.
    Uses rsync over SSH with support for password/key authentication.
    
    Args:
        local_path: Local directory to download TO
        remote_ip: Remote server IP/hostname
        remote_user: Remote username
        remote_auth: {'method': 'password'/'key', 'value': pwd_or_key_path}
        remote_path: Base remote path (subset hierarchy preserved)
        max_retries: Maximum download attempts
    
    Returns:
        True on success, False if all retries fail.
    """
    if not os.path.exists(local_path):
        os.makedirs(local_path, exist_ok=True)
    
    # Compute relative path for remote hierarchy
    try:
        subset_relpath = os.path.relpath(local_path, SAVE_ROOT)
        if subset_relpath.startswith('..') or subset_relpath == '.':
            raise ValueError(f"local_path '{local_path}' is not under SAVE_ROOT '{SAVE_ROOT}'")
    except Exception as e:
        logger.error(f"Failed to compute relative path: {e}")
        subset_relpath = os.path.basename(local_path)
    
    remote_full_path = os.path.join(remote_path, subset_relpath)
    logger.info(f"Download source: {remote_user}@{remote_ip}:{remote_full_path} -> {local_path}")
    
    auth_method: Literal['password', 'key'] = remote_auth.get('method')  # type: ignore
    auth_value = remote_auth.get('value', '')
    
    for attempt in range(1, max_retries + 1):
        logger.info(f"Download attempt {attempt}/{max_retries} for {subset_relpath}")
        error_msg = ""
        
        try:
            # === RSYNC DOWNLOAD ===
            rsync_cmd = ['rsync', '-avz', '--exclude=finish.keep']
            env = None
            
            if auth_method == 'password':
                if not shutil.which('sshpass'):
                    raise RuntimeError("sshpass not found. Install it or use SSH key authentication.")
                rsync_cmd = ['sshpass', '-e'] + rsync_cmd
                rsync_cmd += [
                    '-e', 'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10',
                    f'{remote_user}@{remote_ip}:{remote_full_path.rstrip("/")}/',
                    f'{local_path.rstrip("/")}/'
                ]
                env = os.environ.copy()
                env['SSHPASS'] = auth_value
            else:  # key auth
                if not os.path.exists(auth_value):
                    raise FileNotFoundError(f"SSH key not found: {auth_value}")
                rsync_cmd += [
                    '-e', f'ssh -i {shlex.quote(auth_value)} -o StrictHostKeyChecking=no -o ConnectTimeout=10',
                    f'{remote_user}@{remote_ip}:{remote_full_path.rstrip("/")}/',
                    f'{local_path.rstrip("/")}/'
                ]
            
            # Stream rsync output
            process = subprocess.Popen(
                rsync_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            noise_patterns = {'sending incremental file list', 'building file list', 'total:', 'speedup is', 'bytes/sec'}
            for line in process.stdout or []:
                stripped = line.strip()
                if stripped and not any(pat in stripped for pat in noise_patterns):
                    logger.debug(f"RSYNC-DL: {stripped}")
            
            ret = process.wait(timeout=3600)
            if ret != 0:
                raise RuntimeError(f"RSYNC download failed with exit code {ret}")
            
            logger.info(f"✓ Download succeeded for {subset_relpath} (attempt {attempt})")
            return True
            
        except subprocess.TimeoutExpired as e:
            error_msg = f"Timeout after {e.timeout}s"
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed (exit {e.returncode}): {e.stderr.strip() if e.stderr else 'no output'}"
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
        
        logger.warning(f"Download attempt {attempt} failed for {subset_relpath}: {error_msg}")
        
        if attempt < max_retries:
            backoff = min(2 ** (attempt - 1), 30)
            logger.info(f"Retrying in {backoff}s...")
            time.sleep(backoff)
        else:
            logger.error(f"✗ All {max_retries} download attempts failed for {subset_relpath}")
            return False
    
    return False

def cleanup_info_json(directory: str) -> int:
    """
    Recursively delete all info.json files under directory.
    This allows blender script to re-render objects that were previously completed.
    
    Returns:
        Number of info.json files deleted.
    """
    deleted_count = 0
    for root, dirs, files in os.walk(directory):
        if 'info.json' in files:
            info_path = os.path.join(root, 'info.json')
            try:
                os.remove(info_path)
                deleted_count += 1
                logger.debug(f"Removed: {info_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {info_path}: {e}")
    
    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} info.json files in {directory}")
    return deleted_count


def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    args,
    max_retries = 3
) -> None:
    worker_logger = logging.getLogger(f"worker.gpu{gpu}.{os.getpid()}")
    
    while True:
        item_save = queue.get()
        if item_save is None:
            queue.task_done()
            break

        item, save_path = item_save
        
        # Individual object skip check (info.json)
        if args.skip_exist and os.path.exists(os.path.join(save_path, item, 'info.json')):
            worker_logger.info(f"✓ Skipping {item} (info.json exists)")
            queue.task_done()
            with count.get_lock():
                count.value += 1
            continue

        worker_logger.info(f".Rendering {item} on GPU {gpu}")
        
        command = (
            f"CUDA_VISIBLE_DEVICES={gpu} "
            f"{args.blender} -b -P ./blender/blender_script.py -- "
            f"--object_name {item} "
            f"--output_dir '{save_path}' "
            f"--depth "
            f"--lighting_split {args.split} "
            f"--view_split {args.split} "
        )
        if args.skip_exist:
            command += " --skip_exist"

        start_time = time.time()
        retry_delay = 120
        for attempt in range(1, max_retries + 1):
            start_time = time.time()
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=36000
                )
                duration = time.time() - start_time
                worker_logger.info(f"✓ Rendered {item} in {duration:.1f}s")

                # Log non-frame Blender output
                for line in result.stdout.splitlines():
                    if line.strip() and not line.startswith("Fra:"):
                        worker_logger.debug(f"[{item}] {line}")

                # Success — exit retry loop
                break

            except subprocess.TimeoutExpired:
                duration = time.time() - start_time
                worker_logger.warning(f"✗ TIMEOUT on attempt {attempt}/{max_retries} for {item} after {duration:.1f}s")
                if attempt == max_retries:
                    worker_logger.error(f"✗ FAILED {item} after {max_retries} attempts due to timeout")
                else:
                    worker_logger.info(f"   → Retrying in {retry_delay}s...")

            except subprocess.CalledProcessError as e:
                duration = time.time() - start_time
                worker_logger.warning(f"✗ FAILED on attempt {attempt}/{max_retries} for {item} after {duration:.1f}s (exit {e.returncode})")
                if attempt == max_retries:
                    worker_logger.error(f"✗ FAILED {item} permanently after {max_retries} attempts (final exit code: {e.returncode})")
                else:
                    worker_logger.info(f"   → Retrying in {retry_delay}s...")

            except Exception as e:
                duration = time.time() - start_time
                worker_logger.exception(f"✗ CRITICAL ERROR on attempt {attempt}/{max_retries} rendering {item}: {e}")
                if attempt == max_retries:
                    worker_logger.error(f"✗ ABORTED {item} after {max_retries} failed attempts due to critical error")
                else:
                    worker_logger.info(f"   → Retrying in {retry_delay}s...")

            # Delay before next retry (only if not on final attempt)
            if attempt < max_retries:
                time.sleep(retry_delay)

        with count.get_lock():
            count.value += 1
        queue.task_done()

def rendering(uids, save_path, queue, count, args, subset_idx=0, total_subsets=1, total_objects=0):
    """
    Queue all objects for a subset and wait for completion.
    Returns when all objects in this subset are rendered.
    """
    if args.seed != 0:
        random.seed(args.seed + subset_idx)  # Deterministic per-subset shuffling
        random.shuffle(uids)

    logger.info(f"Queueing {len(uids)} objects for {save_path} (subset {subset_idx+1}/{total_subsets})")
    start_count = count.value
    
    for item in uids:
        queue.put((item, save_path))
    
    # Wait for subset completion with progress reporting
    target_count = start_count + len(uids)
    last_report = start_count
    
    while count.value < target_count:
        current = count.value
        if current > last_report or (current == target_count):
            rendered_this = current - start_count
            progress = rendered_this / len(uids)
            global_progress = current / total_objects if total_objects else 0
            
            logger.info(
                f"Subset {subset_idx+1}/{total_subsets}: {rendered_this}/{len(uids)} ({progress:.1%}) | "
                f"Global: {current}/{total_objects} ({global_progress:.1%})"
            )
            
            if args.log_to_wandb:
                wandb.log({
                    "rendered_count": current,
                    "total_count": total_objects,
                    "subset_progress": progress,
                    "subset_index": subset_idx,
                    "global_progress": global_progress,
                    "current_subset": os.path.basename(save_path),
                })
            
            last_report = current
        
        time.sleep(30)  # Report every 30 seconds
    
    logger.info(f"✓ Subset completed: {len(uids)} objects rendered at {save_path}")
    return len(uids)

def main():
    args = tyro.cli(Args)
    assert args.split in ['training', 'validation', 'testing'], f"Invalid split: {args.split}"
    
    if args.upload:
        args = load_remote_credentials(args)
        logger.warning("⚠️  Using password-based SSH authentication. For production, use SSH keys.")
    
    logger.info("="*70)
    logger.info(f"STARTING RENDERING SESSION | Split: {args.split}")
    logger.info(f"Blender: {args.blender}")
    logger.info(f"Output root: {SAVE_ROOT}")
    logger.info(f"GPUs: {args.gpus} | Workers/GPU: {args.workers_per_gpu}")
    logger.info(f"Skip existing: {args.skip_exist} | Upload: {args.upload}")
    if args.upload:
        logger.info(f"Upload retries: {args.upload_retries} | Remote target: {args.remote_ip}:{args.remote_path}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("="*70)

    if args.log_to_wandb:
        wandb.login(key='f99bf05243fcbfa31ac750bdbc1675282b080eae')
        wandb.init(project="Laval-Objaverse-Dataset-rendering", config=vars(args))
    
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    # Start persistent worker pool
    processes = []
    for gpu_i in args.gpus:
        for _ in range(args.workers_per_gpu):
            p = multiprocessing.Process(
                target=worker,
                args=(queue, count, gpu_i, args)
            )
            p.start()
            processes.append(p)
            logger.info(f"Started worker on GPU {gpu_i} (PID: {p.pid})")

    try:
        # ===== PREPARE WORKLOAD =====
        if args.split == 'training':
            subsets_dir = f"{OBJAVERSE_INFO}/training_subsets"
            subsets = get_subset_json_files(subsets_dir)
            total_objects = 0
            subset_data = []
            
            for subset in subsets:
                with open(os.path.join(subsets_dir, subset), 'r') as f:
                    uids = json.load(f)
                    subset_name = os.path.splitext(subset)[0]
                    save_path = os.path.join(SAVE_ROOT, args.split, subset_name)
                    subset_data.append((subset_name, save_path, uids))
                    total_objects += len(uids)
                    
            logger.info(f"Found {len(subsets)} training subsets ({total_objects} total objects)")
        else:
            with open(f'{OBJAVERSE_INFO}/full_{args.split}_objects.json', 'r') as f:
                uids = json.load(f)
            save_path = os.path.join(SAVE_ROOT, args.split)
            subset_data = [(args.split, save_path, uids)]
            total_objects = len(uids)
            logger.info(f"Processing {args.split} split ({total_objects} objects)")

        # ===== PROCESS EACH SUBSET SEQUENTIALLY =====
        failed_uploads = []
        successful_uploads = []
        
        for subset_idx, (subset_name, save_path, uids) in enumerate(subset_data):
            os.makedirs(save_path, exist_ok=True)
            marker_path = os.path.join(save_path, "finish.keep")
            
            # SKIP CHECK: finish.keep at subset level
            if args.skip_exist and os.path.exists(marker_path):
                logger.info(f"✓ Skipping {subset_name} (finish.keep exists at {marker_path})")
                continue

            # 🔽 NEW: Download from remote if enabled
        if args.download and args.upload:  # download requires remote credentials
            logger.info(f"📥 Downloading {subset_name} from remote before rendering...")
            download_success = download_from_remote(
                save_path,
                args.remote_ip,
                args.remote_user,
                args.remote_auth,
                args.remote_path,
                max_retries=args.upload_retries
            )

            if download_success:
                # Delete info.json to allow re-rendering of completed objects
                cleaned = cleanup_info_json(save_path)
                logger.info(f"✓ Downloaded {subset_name} ({cleaned} info.json files removed for re-render check)")
            else:
                logger.warning(f"⚠️  Download failed for {subset_name}, continuing with local-only rendering...")

                
                logger.info(f"\n{'='*70}")
                logger.info(f"PROCESSING SUBSET {subset_idx+1}/{len(subset_data)}: {subset_name}")
                logger.info(f"Objects: {len(uids)} | Path: {save_path}")
                logger.info('='*70)
            
            # Render the subset
            rendered_count = rendering(
                uids=uids,
                save_path=save_path,
                queue=queue,
                count=count,
                args=args,
                subset_idx=subset_idx,
                total_subsets=len(subset_data),
                total_objects=total_objects
            )
            
            # UPLOAD & CLEANUP (if enabled)
            if args.upload:
                success = upload_and_cleanup(
                    save_path,
                    args.remote_ip,
                    args.remote_user,
                    args.remote_auth,
                    args.remote_path,
                    max_retries=args.upload_retries
                )
                
                if success:
                    successful_uploads.append(subset_name)
                    if args.log_to_wandb:
                        wandb.log({
                            f"upload_success_{subset_name}": 1,
                            "last_uploaded_subset": subset_name,
                            "upload_timestamp": time.time()
                        })
                else:
                    failed_uploads.append(subset_name)
                    logger.error(f"✗ Upload permanently failed for {subset_name} after {args.upload_retries} attempts. Continuing to next subset...")
                    if args.log_to_wandb:
                        wandb.alert(
                            title="Upload Failed",
                            text=f"Subset {subset_name} failed after {args.upload_retries} attempts. Continuing rendering..."
                        )
                    # CRITICAL: Continue to next subset - don't crash entire job
                    continue

        # ===== FINAL SYNCHRONIZATION =====
        logger.info("Waiting for all workers to finish processing queue...")
        queue.join()
        
        # Terminate workers gracefully
        logger.info("Terminating workers...")
        for _ in range(len(args.gpus) * args.workers_per_gpu):
            queue.put(None)
        
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                logger.warning(f"Worker {p.pid} hung - forcing termination")
                p.terminate()
                p.join(5)

        # Final summary
        logger.info("\n" + "="*70)
        logger.info("UPLOAD SUMMARY")
        logger.info(f"Successful uploads: {len(successful_uploads)}")
        if successful_uploads:
            logger.info(f"  {', '.join(successful_uploads[:5])}{'...' if len(successful_uploads) > 5 else ''}")
        logger.info(f"Failed uploads: {len(failed_uploads)}")
        if failed_uploads:
            logger.info(f"  {', '.join(failed_uploads[:5])}{'...' if len(failed_uploads) > 5 else ''}")
        logger.info("="*70)

        if args.log_to_wandb:
            wandb.finish()

    except KeyboardInterrupt:
        logger.error("KeyboardInterrupt received - terminating workers...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        raise
    finally:
        logger.info("\n" + "="*70)
        logger.info(f"SESSION COMPLETE | Total rendered: {count.value}")
        logger.info(f"Log file: {LOG_FILE}")
        logger.info("="*70)

if __name__ == "__main__":
    main()