"""g6547: the trainer's GPU memory per iteration phase, with the caching allocator's free blocks returned after each.

`torch.cuda.max_memory_allocated` was the process's peak since start, so the reports could not say which phase set
it (39.5 GB at g6546 against 11.5 GB held after PPO), and the allocator kept every block it had ever cached (61 GB
reserved). At each phase boundary `Phases.mark` records the phase's own peak allocated and reserved bytes and what
stays allocated, returns the free cached blocks to the driver (`empty_cache`: other processes on the GPU can use
them; the next phase allocates again) and restarts the peak counters. After this the `memory` reports' peak fields
(telemetry.memory) are peaks since the previous boundary, not since the process started.
"""
import time

import torch


class Phases:
    def __init__(self, device):
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda' and torch.cuda.is_available()
        self.phases = {}

    def mark(self, name):
        """The phase that just ended: its peaks since the previous mark and what it leaves allocated; then release."""
        if not self.cuda: return
        d = self.device; began = time.perf_counter()
        entry = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(d),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(d),
                     allocated_bytes=torch.cuda.memory_allocated(d), reserved_bytes=torch.cuda.memory_reserved(d))
        torch.cuda.empty_cache()
        entry.update(reserved_after_release_bytes=torch.cuda.memory_reserved(d), release_seconds=time.perf_counter() - began)
        torch.cuda.reset_peak_memory_stats(d)
        self.phases[name] = entry

    def take(self):
        """This iteration's phases (then cleared), or None off CUDA."""
        if not self.cuda: return None
        phases, self.phases = self.phases, {}
        return dict(phases=phases, scope='Per phase since the previous boundary: peak allocated/reserved bytes, bytes still '
                    'allocated at its end, and reserved bytes after its free cached blocks went back to the driver')
