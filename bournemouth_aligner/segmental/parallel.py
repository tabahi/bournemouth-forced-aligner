"""
The segmental Viterbi in worker processes, for `BFA(n_jobs > 1)`.

Only `align_batch` runs here. It is a pure function of a few CPU tensors, so a worker never
loads a model, a phonemizer or anything else from the aligner: the main process runs the
forward pass and the post-processing, and hands the DP of each batch to the pool.

Workers are started with "spawn" on every platform (the only method macOS and Windows both
have), so a script that sets `n_jobs > 1` must keep its top-level code under
`if __name__ == "__main__":` - each worker re-imports the calling script.

Each job is a WHOLE batch, never a slice of one: `class_prior_normalise` takes its prior over
the batch it is given, so splitting a batch would change the alignment.
"""

import concurrent.futures
import multiprocessing
import weakref

import torch

from .aligner import align_batch


def _init_worker():
    # One DP per worker; the per-frame tensors are far too small to pay for intra-op threads.
    torch.set_num_threads(1)


def align_batch_job(arrays, kwargs):
    """
    `align_batch` on numpy inputs. They travel as numpy rather than tensors because a pickled
    tensor goes through torch's shared-memory reducer, whose behaviour differs per platform.
    """
    tensors = {k: (None if v is None else torch.from_numpy(v)) for k, v in arrays.items()}
    return align_batch(tensors['log_probs'], tensors['seqs'], tensors['seq_lens'],
                       tensors['spectral_lens'], flux=tensors['flux'], **kwargs)


def pack_job(log_probs, seqs, seq_lens, spectral_lens, flux):
    """
    CPU tensors -> numpy arrays for `align_batch_job`, cut to the batch's longest clip. The DP
    never reads a frame past `max(spectral_lens)` (`_segmental_topology`, `path_to_onsets`,
    `refine_onsets` all stop there), so the cut changes nothing but the bytes sent.
    """
    t = max(1, min(int(log_probs.shape[1]), int(spectral_lens.max())))
    arrays = dict(log_probs=log_probs[:, :t], seqs=seqs, seq_lens=seq_lens,
                  spectral_lens=spectral_lens, flux=None if flux is None else flux[:, :t])
    return {k: (None if v is None else v.detach().cpu().contiguous().numpy())
            for k, v in arrays.items()}


def _shutdown(executor):
    try:
        executor.shutdown(wait=True, cancel_futures=True)
    except TypeError:                                   # python 3.8: no cancel_futures
        executor.shutdown(wait=True)


class ViterbiPool:
    """
    A lazily started process pool for `align_batch_job`, shut down when its owner is
    garbage-collected or the interpreter exits.
    """

    def __init__(self, n_jobs):
        self.n_jobs = int(n_jobs)
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.n_jobs, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker)
        self._finalizer = weakref.finalize(self, _shutdown, self._executor)

    def submit(self, arrays, kwargs):
        return self._executor.submit(align_batch_job, arrays, kwargs)

    def close(self):
        self._finalizer()
