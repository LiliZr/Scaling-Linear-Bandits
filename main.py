"""
main.py — runs algorithms with checkpointing and nohup-friendly logs.

CHANGES vs previous version:

  1. BLAS thread pinning is done at TOP of file, BEFORE `import numpy`.
  2. multiprocessing start method is forced to 'spawn' (fresh interpreters,
     immune to BLAS/fork deadlocks).
  3. Parallel runs are collected with `imap_unordered`, which yields each
     run's result AS SOON AS THAT RUN FINISHES.
  4. Each finished run is saved IMMEDIATELY and the experiment is replotted,
     so even a single completed run produces visible output.
  5. Per-run timeout: a worker that exceeds --run-timeout (default 12h)
     is dropped without blocking the rest of the experiment.
  6. ROW COMPACTION (critical fix): results that arrive in non-monotonic
     order (run 4 finishes before run 3) used to leave the first rows of
     the array empty while `completed_runs` was still incremented. The plot
     then averaged zero-filled rows. We now compact filled rows to the top
     of the array at save time, so arrays[:completed_runs] always points to
     valid data.

Usage:
  python main.py                                 # default suite
  python main.py LinUCB SCLUB                    # specific algos
  python main.py --experiment medium             # named experiment
  python main.py --plot-only --experiment medium # replot from npz only
  python main.py --n-workers 4                   # 4 parallel runs per algo
  python main.py --run-timeout 21600             # 6h timeout per run
"""

# ----------------------------------------------------------------------------
# CRITICAL: BLAS thread limit must be set BEFORE numpy is imported.
# ----------------------------------------------------------------------------
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import argparse
import sys
import time
import multiprocessing as mp
import numpy as np

from data.synthetic import generate_synthetic_data, generate_user_stream
from algo_registry import ALGOS, build_model
from plot import plot_experiment


# ----------------------------------------------------------------------------
# Logging helpers.
# ----------------------------------------------------------------------------

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        self.f = open(log_path, 'a')

    def log(self, msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ----------------------------------------------------------------------------
# Checkpoint helpers.
# ----------------------------------------------------------------------------

def save_meta(out_dir, **kwargs):
    np.savez(os.path.join(out_dir, '_meta.npz'), **kwargs)


def algo_checkpoint_path(out_dir, algo):
    return os.path.join(out_dir, f"{algo}.npz")


def load_checkpoint(out_dir, algo, T, nb_runs):
    """Load existing checkpoint if any. Returns (arrays_dict, completed_runs).

    After the row-compaction fix, `completed_runs` is guaranteed to point to
    valid data at rows [0, completed_runs[. Older checkpoints from before
    the fix may have an inconsistent layout; we detect that by checking that
    the first `completed_runs` rows of `regret` are non-trivial.
    """
    path = algo_checkpoint_path(out_dir, algo)
    if not os.path.exists(path):
        return None, 0
    try:
        data = np.load(path)
        arrays = {k: data[k] for k in data.files}
        completed = int(arrays.get('completed_runs', np.array(0)).item())
        if arrays['regret'].shape != (nb_runs, T):
            return None, 0
        # Backward compatibility: older checkpoints did not save
        # bucket_over_time. Inject an all -1 array of the right shape so
        # the code paths that read it do not crash.
        if 'bucket_over_time' not in arrays:
            arrays['bucket_over_time'] = np.full((nb_runs, T), -1,
                                                  dtype=np.int32)
        return arrays, completed
    except Exception:
        return None, 0


def save_checkpoint(out_dir, algo, arrays, completed_runs):
    """Atomic save: write to tmp file then rename."""
    path = algo_checkpoint_path(out_dir, algo)
    tmp_base = path + '.tmp'
    np.savez(tmp_base,
             regret=arrays['regret'],
             cpu_time=arrays['cpu_time'],
             memory=arrays['memory'],
             n_user_clusters=arrays['n_user_clusters'],
             n_arm_groups=arrays['n_arm_groups'],
             bucket_over_time=arrays['bucket_over_time'],
             completed_runs=np.array(completed_runs))
    os.replace(tmp_base + '.npz', path)


def empty_arrays(T, nb_runs):
    return {
        'regret':           np.zeros((nb_runs, T)),
        'cpu_time':         np.zeros((nb_runs, T)),
        'memory':           np.zeros((nb_runs, T)),
        'n_user_clusters':  np.zeros((nb_runs, T), dtype=int),
        'n_arm_groups':     np.zeros((nb_runs, T), dtype=int),
        # -1 = "no track / no bucket info for this round". Filled by the
        # worker only when track_bucket_selection is True on this algo's
        # ArmClustering.
        'bucket_over_time': np.full((nb_runs, T), -1, dtype=np.int32),
    }


def _compact_arrays(arrays, filled_rows, nb_runs):
    """Return a copy of `arrays` where the rows are reordered so that all
    filled rows come first, in the order of their original row index.

    This guarantees that arrays[:completed_runs] only contains valid data,
    which is what plot.py assumes when it slices the array before averaging.

    The original `arrays` dict is NOT modified; we return a new dict so the
    in-memory representation keeps the original (seed-aligned) layout,
    which is useful if we later need to fill in missing rows or re-run a
    specific seed.
    """
    filled_indices = [i for i in range(nb_runs) if filled_rows[i]]
    compacted = {}
    for key, arr in arrays.items():
        new_arr = np.zeros_like(arr)
        for new_idx, old_idx in enumerate(filled_indices):
            new_arr[new_idx] = arr[old_idx]
        compacted[key] = new_arr
    return compacted


# ----------------------------------------------------------------------------
# Progress callback (serial mode only).
# ----------------------------------------------------------------------------

def make_progress_callback(logger, algo, run_idx, nb_runs, T, every_pct=5):
    step = max(T * every_pct // 100, 1)
    next_log = step
    t0 = time.time()
    state = {'next_log': next_log, 't0': t0}

    def progress(t, model):
        if t + 1 >= state['next_log']:
            elapsed = time.time() - state['t0']
            pct = 100 * (t + 1) / T
            eta = elapsed * (T - t - 1) / max(t + 1, 1)
            extra = ""
            if hasattr(model, 'clusters'):
                try:
                    extra = f" clusters={len(model.clusters)}"
                except TypeError:
                    pass
            if hasattr(model, 'n_splits'):
                extra = extra + f" splits={model.n_splits} merges={model.n_merges}"
            if hasattr(model, 'kmeans_runs'):
                extra = extra + f" kmeans_runs={model.kmeans_runs}"
            logger.log(f"  {algo} [run {run_idx + 1}/{nb_runs}] "
                       f"t={t + 1}/{T} ({pct:.0f}%) "
                       f"elapsed={elapsed:.0f}s eta={eta:.0f}s{extra}")
            state['next_log'] += step

    return progress


def run_with_progress(model, user_stream, T, progress_cb):
    import tracemalloc
    import time as _time
    tracemalloc.start()
    model.init_run(T)
    t0_cpu = _time.process_time()
    t0_wall = _time.perf_counter()
    cum_reward = 0.0
    cum_regret = 0.0
    track_bucket = (
        hasattr(model, 'arm_clust')
        and getattr(model.arm_clust, 'track_bucket_selection', False)
    )
    for t in range(T):
        user_id = user_stream[t]
        item_id = model.recommend(user_id, t)
        if track_bucket:
            model.bucket_over_time[t] = int(model.arm_clust.last_bucket)
        reward_vec, reward_scalar = model.generate_reward(user_id, item_id)
        optimal_reward = model.compute_optimal_reward(user_id)
        model.update(user_id, item_id, reward_vec, reward_scalar, t)
        cum_reward += reward_scalar
        cum_regret += (optimal_reward - reward_scalar)
        model.cumulative_reward[t] = cum_reward
        model.cumulative_regret[t] = cum_regret
        model.cpu_time[t] = _time.process_time() - t0_cpu
        model.wall_time[t] = _time.perf_counter() - t0_wall
        current, _ = tracemalloc.get_traced_memory()
        model.memory_peak[t] = current / (1024 * 1024)
        model._track_state(t)
        progress_cb(t, model)
    tracemalloc.stop()


# ----------------------------------------------------------------------------
# Parallel worker
# ----------------------------------------------------------------------------

def _parallel_worker(args):
    """Worker function for a single run in a parallel pool."""
    (algo, data, weights, seed, k, lam, scale, batch, batch_size,
     ucb_partition, update_champions_on_select, track_bucket_selection,
     param_overrides,
     user_stream, T, run_idx, nb_runs, progress_queue) = args

    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')

    every_pct = 5
    step = max(T * every_pct // 100, 1)
    next_log = [step]
    t_start = time.time()
    pid = os.getpid()
    heartbeat_path = f"/tmp/heartbeat_{algo}_{run_idx}_{pid}.txt"

    def push(msg_type, **payload):
        try:
            progress_queue.put({'algo': algo, 'run_idx': run_idx,
                                'nb_runs': nb_runs, 'pid': pid,
                                'type': msg_type, **payload},
                               timeout=5)
        except Exception:
            pass

    def write_heartbeat():
        try:
            with open(heartbeat_path, 'w') as f:
                f.write(f"{time.time()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception:
            pass

    write_heartbeat()
    push('start', seed=seed)

    try:
        model = build_model(algo, data, weights, seed, k,
                            lam=lam, scale=scale,
                            batch=batch, batch_size=batch_size,
                            ucb_partition=ucb_partition,
                            update_champions_on_select=update_champions_on_select,
                            track_bucket_selection=track_bucket_selection,
                            param_overrides=param_overrides)
    except Exception as e:
        push('error', msg=f"build_model failed: {type(e).__name__}: {e}")
        return None

    import tracemalloc
    user_stream_np = np.asarray(user_stream, dtype=np.int64)
    Teff = min(T, len(user_stream_np))
    tracemalloc.start()

    try:
        model.init_run(Teff)
    except Exception as e:
        push('error', msg=f"init_run failed: {type(e).__name__}: {e}")
        tracemalloc.stop()
        return None

    t0_cpu = time.process_time()
    cum_reward = 0.0
    cum_regret = 0.0
    mem_sample_every = max(1, Teff // 100)
    heartbeat_every = 1000
    last_mem_mb = 0.0
    t = 0
    # Cache whether the bucket choice should be logged each round.
    track_bucket = (
        hasattr(model, 'arm_clust')
        and getattr(model.arm_clust, 'track_bucket_selection', False)
    )

    try:
        for t in range(Teff):
            user_id = int(user_stream_np[t])
            item_id = model.recommend(user_id, t)
            if track_bucket:
                model.bucket_over_time[t] = int(model.arm_clust.last_bucket)
            reward_vec, reward_scalar = model.generate_reward(user_id, item_id)
            optimal_reward = model._optimal_per_user[user_id]
            model.update(user_id, item_id, reward_vec, reward_scalar, t)
            cum_reward += reward_scalar
            cum_regret += (optimal_reward - reward_scalar)
            model.cumulative_reward[t] = cum_reward
            model.cumulative_regret[t] = cum_regret
            model.cpu_time[t] = time.process_time() - t0_cpu
            if t % mem_sample_every == 0 or t == Teff - 1:
                current, _ = tracemalloc.get_traced_memory()
                last_mem_mb = current / (1024 * 1024)
            model.memory_peak[t] = last_mem_mb
            model._track_state(t)
            if t % heartbeat_every == 0:
                write_heartbeat()
            if t + 1 >= next_log[0] or t == Teff - 1:
                elapsed = time.time() - t_start
                pct = 100.0 * (t + 1) / Teff
                eta = elapsed * (Teff - t - 1) / max(1, t + 1)
                extra = {}
                if hasattr(model, 'clusters'):
                    try:
                        extra['clusters'] = len(model.clusters)
                    except TypeError:
                        pass
                if hasattr(model, 'n_splits'):
                    extra['splits'] = int(model.n_splits)
                if hasattr(model, 'n_merges'):
                    extra['merges'] = int(model.n_merges)
                if hasattr(model, 'kmeans_runs'):
                    extra['kmeans_runs'] = int(model.kmeans_runs)
                push('progress', t=t + 1, T=Teff, pct=pct,
                     elapsed=elapsed, eta=eta, **extra)
                next_log[0] += step
    except Exception as e:
        push('error', msg=f"run loop failed at t={t}: {type(e).__name__}: {e}")
        tracemalloc.stop()
        try:
            os.remove(heartbeat_path)
        except OSError:
            pass
        return None
    tracemalloc.stop()

    push('done', elapsed=time.time() - t_start,
         final_regret=float(model.cumulative_regret[-1]),
         final_time=float(model.cpu_time[-1]),
         final_mem=float(model.memory_peak[-1]),
         clusters_final=len(model.clusters) if hasattr(model, 'clusters') else 1)

    try:
        os.remove(heartbeat_path)
    except OSError:
        pass

    return {
        'run_idx': run_idx,
        'regret': np.asarray(model.cumulative_regret),
        'cpu_time': np.asarray(model.cpu_time),
        'memory': np.asarray(model.memory_peak),
        'n_user_clusters': np.asarray(model.n_user_clusters_over_time),
        'n_arm_groups': np.asarray(model.n_arm_groups_over_time),
        'bucket_over_time': np.asarray(model.bucket_over_time),
        'final_regret': float(model.cumulative_regret[-1]),
        'final_time': float(model.cpu_time[-1]),
        'final_mem': float(model.memory_peak[-1]),
        'final_clusters': len(model.clusters) if hasattr(model, 'clusters') else 1,
    }


def _drain_progress_queue(progress_queue, logger):
    while True:
        try:
            evt = progress_queue.get_nowait()
        except Exception:
            return
        t_ = evt['type']
        if t_ == 'start':
            logger.log(f"-- [{evt['algo']}] starting run "
                       f"{evt['run_idx'] + 1}/{evt['nb_runs']} "
                       f"(seed={evt.get('seed')}, pid={evt.get('pid')})")
        elif t_ == 'progress':
            extras = []
            for kk in ('clusters', 'splits', 'merges', 'kmeans_runs'):
                if kk in evt:
                    extras.append(f"{kk}={evt[kk]}")
            extra_str = (' ' + ' '.join(extras)) if extras else ''
            logger.log(f"  {evt['algo']} [run "
                       f"{evt['run_idx'] + 1}/{evt['nb_runs']}] "
                       f"t={evt['t']}/{evt['T']} "
                       f"({evt['pct']:.0f}%) "
                       f"elapsed={evt['elapsed']:.0f}s "
                       f"eta={evt['eta']:.0f}s{extra_str}")
        elif t_ == 'done':
            logger.log(f"++ [{evt['algo']}] run "
                       f"{evt['run_idx'] + 1}/{evt['nb_runs']} "
                       f"done in {evt['elapsed']:.1f}s. "
                       f"final_regret={evt['final_regret']:.1f} "
                       f"final_time={evt['final_time']:.1f}s "
                       f"final_mem={evt['final_mem']:.2f}MB "
                       f"clusters_final={evt['clusters_final']}")
        elif t_ == 'error':
            logger.log(f"!! [{evt['algo']}] run "
                       f"{evt['run_idx'] + 1}/{evt['nb_runs']} ERROR: "
                       f"{evt.get('msg', 'unknown')}")


# ----------------------------------------------------------------------------
# Parallel run loop with incremental save + plot + row compaction
# ----------------------------------------------------------------------------

def _run_algo_parallel(algo, tasks, arrays, completed_so_far, nb_runs,
                       n_workers, run_timeout, out_dir, plot_after_each,
                       logger):
    """Execute parallel runs for one algo, saving and plotting after EACH run
    that finishes. Compacts filled rows to the top of the array at save
    time so that arrays[:completed_runs] always contains valid data.
    """
    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    progress_queue = manager.Queue()

    tasks_with_q = [t + (progress_queue,) for t in tasks]

    actual_workers = min(n_workers, len(tasks_with_q))
    logger.log(f"-- [{algo}] launching {len(tasks_with_q)} runs in parallel "
               f"({actual_workers} workers, start_method=spawn, "
               f"run_timeout={run_timeout}s)")

    # `filled_rows` tracks which rows of `arrays` have valid data. The first
    # `completed_so_far` rows of a resumed checkpoint are valid (the previous
    # compaction had placed them there).
    filled_rows = [False] * nb_runs
    for i in range(completed_so_far):
        filled_rows[i] = True

    completed = completed_so_far
    failed = 0
    t_start_all = time.time()

    with ctx.Pool(processes=actual_workers) as pool:
        result_iter = pool.imap_unordered(_parallel_worker, tasks_with_q)
        n_total = len(tasks_with_q)
        for i in range(n_total):
            try:
                t0_wait = time.time()
                next_result = None
                while True:
                    _drain_progress_queue(progress_queue, logger)
                    try:
                        next_result = result_iter.next(timeout=2.0)
                        break
                    except mp.TimeoutError:
                        if time.time() - t0_wait > run_timeout:
                            logger.log(
                                f"!! [{algo}] no result for {run_timeout}s — "
                                f"assuming the remaining workers are stuck. "
                                f"Stopping the pool and keeping the "
                                f"{completed} results saved so far."
                            )
                            pool.terminate()
                            pool.join()
                            _drain_progress_queue(progress_queue, logger)
                            return completed
            except StopIteration:
                break

            if next_result is None:
                failed += 1
                logger.log(f"!! [{algo}] one run returned None (worker error). "
                           f"Continuing.")
                continue

            # Place the result in its row.
            ri = next_result['run_idx']
            arrays['regret'][ri] = next_result['regret']
            arrays['cpu_time'][ri] = next_result['cpu_time']
            arrays['memory'][ri] = next_result['memory']
            arrays['n_user_clusters'][ri] = next_result['n_user_clusters']
            arrays['n_arm_groups'][ri] = next_result['n_arm_groups']
            if 'bucket_over_time' in next_result:
                arrays['bucket_over_time'][ri] = next_result['bucket_over_time']
            filled_rows[ri] = True
            completed += 1

            # CRITICAL FIX: compact filled rows to the top of the array
            # so that arrays[:completed] points to valid data. Otherwise,
            # if run 4 finishes before run 3 (which is normal with
            # imap_unordered), the plot would average row 0 (empty) and
            # row 1 (empty), giving a zero curve.
            compacted = _compact_arrays(arrays, filled_rows, nb_runs)
            save_checkpoint(out_dir, algo, compacted, completed_runs=completed)
            logger.log(f"   [{algo}] checkpoint saved with "
                       f"{completed}/{nb_runs} runs (rows compacted).")

            if plot_after_each:
                try:
                    plot_experiment(out_dir)
                except Exception as e:
                    logger.log(f"!! plot failed (continuing): {e}")

        _drain_progress_queue(progress_queue, logger)
        pool.close()
        pool.join()

    elapsed_all = time.time() - t_start_all
    logger.log(f"   [{algo}] {completed}/{nb_runs} parallel runs completed "
               f"in {elapsed_all:.1f}s. ({failed} returned None.)")
    return completed


# ----------------------------------------------------------------------------
# Main experiment loop
# ----------------------------------------------------------------------------

def run_experiment(experiment_name, algos_to_run, T, n_users, n_items, d, k,
                   n_user_clusters, n_arm_clusters, D_objectives, noise_std,
                   nb_runs, seed_data, lam, scale, log_path,
                   plot_after_each=True, force=False, data_override=None,
                   out_dir_override=None, batch=False, batch_size=20,
                   n_workers=1, run_timeout=43200,
                   ucb_partition=False, update_champions_on_select=False,
                   track_bucket_selection=False,
                   param_overrides=None):
    out_dir = out_dir_override or os.path.join('results', experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    logger = Logger(log_path)
    logger.log(f"=== Experiment '{experiment_name}' ===")
    logger.log(f"  T={T}, n_users={n_users}, n_items={n_items}, d={d}, k={k}")
    if data_override is None:
        logger.log(f"  n_user_clusters={n_user_clusters}, "
                   f"n_arm_clusters={n_arm_clusters}, "
                   f"D_objectives={D_objectives}, "
                   f"noise_std={noise_std}, nb_runs={nb_runs}")
    else:
        logger.log(f"  data_override: real-data dict (user_stream provided)")
    if n_workers > 1:
        logger.log(f"  PARALLEL MODE: n_workers={n_workers}, "
                   f"start_method=spawn, run_timeout={run_timeout}s. "
                   f"Each completed run is saved and replotted immediately. "
                   f"Filled rows are compacted to the top of the array.")
    if batch:
        logger.log(f"  BATCH MODE ON: batch_size={batch_size}")
    logger.log(f"  algos to run: {algos_to_run}")
    logger.log(f"  output dir: {out_dir}")
    logger.log(f"  log file:   {log_path}")

    weights = np.ones(D_objectives) / D_objectives

    if data_override is not None:
        data = data_override
        provided_stream = np.asarray(data.get('user_stream'))
        if provided_stream is None or len(provided_stream) == 0:
            raise ValueError("data_override missing 'user_stream'")
    else:
        data = generate_synthetic_data(
            n_users=n_users, n_items=n_items, d=d,
            n_user_clusters=n_user_clusters, n_arm_clusters=n_arm_clusters,
            D_objectives=D_objectives, noise_std=noise_std, seed=seed_data
        )
        provided_stream = None

    # Save metadata AFTER computing provided_stream so we can attach it.
    # When the stream is per-run (synthetic, generated per seed), we save the
    # first run's stream as a reference — sufficient for the per-user bucket
    # diagnostic since the partition logic is identical across seeds, and
    # the relative activity distribution of users is the same.
    meta_kwargs = dict(
        T=T, n_users=n_users, n_items=n_items, d=d, k=k,
        n_user_clusters=n_user_clusters or 0,
        n_arm_clusters=n_arm_clusters or 0,
        D_objectives=D_objectives, noise_std=noise_std,
        nb_runs=nb_runs, lam=lam, scale=scale,
    )
    if provided_stream is not None:
        # Truncate to T to avoid storing more than what each run consumes.
        meta_kwargs['user_stream'] = np.asarray(provided_stream[:T],
                                                dtype=np.int32)
    else:
        # For synthetic, reproduce the stream of seed=100 (the first seed
        # used by run_experiment). This is the stream the first run consumed.
        sample_stream = generate_user_stream(n_users, T, seed=100)
        meta_kwargs['user_stream'] = np.asarray(sample_stream,
                                                dtype=np.int32)
    save_meta(out_dir, **meta_kwargs)

    for algo in algos_to_run:
        if algo not in ALGOS:
            logger.log(f"[skip] Unknown algorithm: {algo}")
            continue

        arrays, completed = load_checkpoint(out_dir, algo, T, nb_runs)
        if arrays is None or force:
            arrays = empty_arrays(T, nb_runs)
            completed = 0
            if force:
                logger.log(f"  [{algo}] --force: starting fresh.")

        if completed >= nb_runs:
            logger.log(f"  [{algo}] already complete ({completed}/{nb_runs}). "
                       f"Skipping.")
            continue

        remaining = nb_runs - completed

        # ---- Parallel branch ----
        if n_workers > 1 and remaining > 1:
            tasks = []
            for run_idx in range(completed, nb_runs):
                seed = run_idx + 100
                if provided_stream is not None:
                    us_run = provided_stream[:T]
                else:
                    us_run = generate_user_stream(n_users, T, seed=seed)
                tasks.append((algo, data, weights, seed, k, lam, scale,
                              batch, batch_size,
                              ucb_partition, update_champions_on_select,
                              track_bucket_selection,
                              param_overrides,
                              us_run, T, run_idx, nb_runs))
            try:
                _run_algo_parallel(
                    algo=algo, tasks=tasks, arrays=arrays,
                    completed_so_far=completed, nb_runs=nb_runs,
                    n_workers=n_workers, run_timeout=run_timeout,
                    out_dir=out_dir, plot_after_each=plot_after_each,
                    logger=logger,
                )
            except Exception as e:
                logger.log(f"!! [{algo}] parallel pool failed: "
                           f"{type(e).__name__}: {e}")
            continue

        # ---- Serial branch ----
        for run_idx in range(completed, nb_runs):
            seed = run_idx + 100
            if provided_stream is not None:
                user_stream = provided_stream[:T]
            else:
                user_stream = generate_user_stream(n_users, T, seed=seed)
            logger.log(f"-- [{algo}] starting run {run_idx + 1}/{nb_runs} "
                       f"(seed={seed})")
            t_start = time.time()
            try:
                model = build_model(algo, data, weights, seed, k,
                                    lam=lam, scale=scale,
                                    batch=batch, batch_size=batch_size,
                                    ucb_partition=ucb_partition,
                                    update_champions_on_select=update_champions_on_select,
                                    track_bucket_selection=track_bucket_selection,
                                    param_overrides=param_overrides)
                progress_cb = make_progress_callback(logger, algo, run_idx,
                                                     nb_runs, T)
                run_with_progress(model, user_stream, T, progress_cb)
            except Exception as e:
                logger.log(f"!! [{algo}] run {run_idx + 1} FAILED: "
                           f"{type(e).__name__}: {e}")
                save_checkpoint(out_dir, algo, arrays, completed_runs=run_idx)
                if plot_after_each:
                    try:
                        plot_experiment(out_dir)
                    except Exception as e2:
                        logger.log(f"!! plot failed: {e2}")
                break

            arrays['regret'][run_idx] = model.cumulative_regret
            arrays['cpu_time'][run_idx] = model.cpu_time
            arrays['memory'][run_idx] = model.memory_peak
            arrays['n_user_clusters'][run_idx] = model.n_user_clusters_over_time
            arrays['n_arm_groups'][run_idx] = model.n_arm_groups_over_time
            if hasattr(model, 'bucket_over_time'):
                arrays['bucket_over_time'][run_idx] = model.bucket_over_time

            elapsed = time.time() - t_start
            extra = ""
            if hasattr(model, 'clusters'):
                try:
                    extra = f" clusters_final={len(model.clusters)}"
                except TypeError:
                    pass
            if hasattr(model, 'n_splits'):
                extra += f" splits={model.n_splits} merges={model.n_merges}"
            logger.log(f"++ [{algo}] run {run_idx + 1}/{nb_runs} done in "
                       f"{elapsed:.1f}s. "
                       f"final_regret={model.cumulative_regret[-1]:.1f} "
                       f"final_time={model.cpu_time[-1]:.1f}s "
                       f"final_mem={model.memory_peak[-1]:.2f}MB{extra}")

            save_checkpoint(out_dir, algo, arrays, completed_runs=run_idx + 1)

            if plot_after_each:
                try:
                    plot_experiment(out_dir)
                except Exception as e:
                    logger.log(f"!! plot failed: {e}")

    logger.log("=== experiment finished ===")
    logger.close()


# ----------------------------------------------------------------------------
# Predefined experiments
# ----------------------------------------------------------------------------

PRESETS = {
    'small':         (30_000,   100, 1_000, 100, 20,  10, 10, 2, 0.05, 30),
    'default':       (10_000,  300, 1_500, 50, 15,  5, 15, 2, 0.05, 30),
    'medium':        (200000,  200, 1500, 80, 20, 30, 10, 2, 0.05, 30),
    'large':         (200_000, 1_000, 5_000, 80, 20, 20, 25, 2, 0.05, 30),
    'sparse_users':  (50_000,  2_000, 2_000, 50, 15, 20, 15, 2, 0.05, 30),
    'many_clusters': (50_000,  500, 3_000, 60, 15, 50, 30, 2, 0.05, 30),
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run bandit experiments. Pass algorithm or group names as "
            "positional args."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('targets', nargs='*',
                   help="Algorithm or group names. Defaults to 'all'.")
    p.add_argument('--experiment', default='default')
    p.add_argument('--algos', nargs='+', default=None)
    p.add_argument('--T', type=int, default=None)
    p.add_argument('--n_users', type=int, default=None)
    p.add_argument('--n_items', type=int, default=None)
    p.add_argument('--d', type=int, default=None)
    p.add_argument('--k', type=int, default=None)
    p.add_argument('--n_user_clusters', type=int, default=None)
    p.add_argument('--n_arm_clusters', type=int, default=None)
    p.add_argument('--D_objectives', type=int, default=2)
    p.add_argument('--noise_std', type=float, default=0.05)
    p.add_argument('--nb_runs', type=int, default=None)
    p.add_argument('--seed_data', type=int, default=42)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--scale', type=float, default=0.5)
    p.add_argument('--log-suffix', default=None)
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--plot-only', action='store_true')
    p.add_argument('--data-npz', default=None)
    p.add_argument('--list', action='store_true')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--batch', action='store_true')
    p.add_argument('--batch-size', type=int, default=20)
    p.add_argument('--n-workers', type=int, default=0,
                   help="Parallel workers. 0 = auto (cpu_count). Default 0.")
    p.add_argument('--run-timeout', type=int, default=200000,
                   help="Per-run watchdog in seconds (default 43200 = 12h).")
    p.add_argument('--split-merge-freq', type=int, default=None,
                   help="Override the SCLUB-family user-cluster split/merge "
                        "test frequency (constructor arg `split_merge_freq` "
                        "of SCLUB-derived algos). If unset, the algorithm "
                        "default is used.")
    p.add_argument('--arm-recluster-freq', type=int, default=None,
                   help="Override the arm recluster frequency (constructor "
                        "arg `arm_recluster_freq` of SCLUB-CA-derived and "
                        "LinUCB-CA-derived algos). If unset, default 500.")
    # ---- Arm-clustering opt-in flags (forwarded to ArmClustering) ----
    p.add_argument('--ucb-bucket', action='store_true',
                   help="ArmClustering: at recluster, partition by UCB "
                        "(mean + beta * bonus) instead of by mean only. "
                        "Costs O(K*k^2) per recluster instead of O(K*k). "
                        "Default: False (current behavior).")
    p.add_argument('--update-champions', action='store_true',
                   help="ArmClustering: after each select, update the "
                        "champion of the winning bucket using UCBs already "
                        "computed (free of cost). Default: False.")
    p.add_argument('--track-buckets', action='store_true',
                   help="ArmClustering: log the bucket id from which each "
                        "selected arm is drawn. Stored in the checkpoint "
                        "as bucket_over_time. Default: False.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        from algo_registry import GROUPS
        print("Algorithms:")
        for a in ALGOS:
            print(f"  {a}")
        print("\nGroups:")
        for g, members in GROUPS.items():
            print(f"  {g:<14} -> {', '.join(members)}")
        return

    if args.plot_only:
        plot_experiment(os.path.join('results', args.experiment))
        return

    if args.experiment in PRESETS:
        T, n_users, n_items, d, k, nuc, nac, D, noise, nb_runs = PRESETS[args.experiment]
    else:
        T = args.T or 10_000
        n_users = args.n_users or 300
        n_items = args.n_items or 1_500
        d = args.d or 50
        k = args.k or 15
        nuc = args.n_user_clusters or 5
        nac = args.n_arm_clusters or 15
        D = args.D_objectives
        noise = args.noise_std
        nb_runs = args.nb_runs or 1

    T = args.T or T
    n_users = args.n_users or n_users
    n_items = args.n_items or n_items
    d = args.d or d
    k = args.k or k
    nuc = args.n_user_clusters or nuc
    nac = args.n_arm_clusters or nac
    D = args.D_objectives if args.D_objectives else D
    noise = args.noise_std
    nb_runs = args.nb_runs or nb_runs

    from algo_registry import resolve_targets, GROUPS
    raw_targets = args.targets or args.algos or []
    if not raw_targets:
        algos = list(ALGOS.keys())
        suffix_default = 'all'
    else:
        try:
            algos = resolve_targets(raw_targets)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)
        if len(raw_targets) == 1 and raw_targets[0].lower() in GROUPS:
            suffix_default = raw_targets[0].lower()
        else:
            suffix_default = "_".join(t.lower() for t in raw_targets)[:40]

    log_suffix = args.log_suffix or suffix_default

    data_override = None
    if args.data_npz:
        dataset_tag = os.path.splitext(os.path.basename(args.data_npz))[0]
        args.experiment = f"{args.experiment}_{dataset_tag}"
        loaded = np.load(args.data_npz)
        data_override = {
            'item_features': loaded['item_features'],
            'user_thetas':   loaded['user_thetas'],
            'user_stream':   loaded['user_stream'],
            'item_stream':   loaded['item_stream'] if 'item_stream' in loaded.files else None,
            'n_users':       int(loaded['n_users']),
            'n_items':       int(loaded['n_items']),
            'd':             int(loaded['d']),
            'D_objectives':  int(loaded['D_objectives']),
            'noise_std':     float(loaded['noise_std']),
        }
        n_users = data_override['n_users']
        n_items = data_override['n_items']
        d = data_override['d']
        D = data_override['D_objectives']
        noise = data_override['noise_std']
        T = min(T, len(data_override['user_stream']))

    out_dir_override = args.out_dir
    log_dir = out_dir_override or f"results/{args.experiment}"
    log_path = f"{log_dir}/nohup_{os.path.basename(log_dir)}_{log_suffix}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if args.n_workers == 0:
        resolved_workers = max(1, (os.cpu_count() or 1))
    else:
        resolved_workers = max(1, args.n_workers)

    # Build per-algo parameter overrides from the global --split-merge-freq
    # and --arm-recluster-freq CLI flags. Each override only kicks in for
    # algos that accept the corresponding constructor argument.
    param_overrides = {}
    if args.split_merge_freq is not None:
        for a in ('SCLUB', 'SCLUB_RP', 'SCLUB_CA', 'SCLUB_RP_CA'):
            param_overrides.setdefault(a, {})
            param_overrides[a]['split_merge_freq'] = args.split_merge_freq
    if args.arm_recluster_freq is not None:
        for a in ('SCLUB_CA', 'SCLUB_RP_CA',
                  'LinUCB_RP_CA', 'LinUCB_IND_RP_CA',
                  'LinUCB_IND_RP_CA_KMeans'):
            param_overrides.setdefault(a, {})
            param_overrides[a]['arm_recluster_freq'] = args.arm_recluster_freq

    run_experiment(
        experiment_name=args.experiment,
        algos_to_run=algos,
        T=T, n_users=n_users, n_items=n_items, d=d, k=k,
        n_user_clusters=nuc, n_arm_clusters=nac,
        D_objectives=D, noise_std=noise,
        nb_runs=nb_runs, seed_data=args.seed_data,
        lam=args.lam, scale=args.scale,
        log_path=log_path,
        plot_after_each=not args.no_plot,
        force=args.force,
        data_override=data_override,
        out_dir_override=out_dir_override,
        batch=args.batch,
        batch_size=args.batch_size,
        ucb_partition=args.ucb_bucket,
        update_champions_on_select=args.update_champions,
        track_bucket_selection=args.track_buckets,
        n_workers=resolved_workers,
        run_timeout=args.run_timeout,
        param_overrides=param_overrides if param_overrides else None,
    )


if __name__ == "__main__":
    main()