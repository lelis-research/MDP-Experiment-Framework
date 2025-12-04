from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter 
import os
import json



class JsonlCallback:
    """
    Super-fast logger that writes one JSON record per call:

        {
          "step": ...,
          "tag": "...",
          "metrics": {"key": value, ...}
        }

    Designed for compute clusters: extremely lightweight,
    no TensorBoard penalty during training.
    """

    def __init__(self, log_path, flush_every=100):
        """
        Args:
            log_path: path to metrics.jsonl
            flush_every: flush OS buffers every N normal writes (force writes flush immediately)
        """
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        self.flush_every = int(flush_every)

        # Line-buffered file (buffering=1): flush on newline by default
        self._f = open(self.log_path, "a", buffering=1)
        self._n_written = 0

    def __call__(self, data_dict, tag, counter, force=False):
        """
        Args:
            data_dict: {metric_name: value}
            tag:       TB-style tag ("agents/run_0")
            counter:   x-axis step (int)
            force:     write immediately, do not increment counters
        """
        record = {
            "step": int(counter),
            "tag": str(tag),
            "metrics": {k: float(v) for k, v in data_dict.items()},
        }

        # Write record
        self._f.write(json.dumps(record) + "\n")

        if force:
            # Force this single record to disk immediately
            self._f.flush()
            return

        # Normal (non-force) logging
        self._n_written += 1

        if self.flush_every > 0 and (self._n_written % self.flush_every == 0):
            self._f.flush()

    def reset(self):
        """Reset only the internal write counter; file stays open."""
        self._n_written = 0

    def close(self):
        """Flush pending writes and close the file."""
        if self._f and not self._f.closed:
            self._f.flush()
            self._f.close()

class TBCallBack:
    def __init__(self, log_dir, flush_every=1000):
        """
        Parameters
        ----------
        log_dir : str
            Directory for TensorBoard logs.
        flush_every : int
            Number of __call__ invocations between flushes of the averaged data.
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_counter = 0
        self.flush_every = flush_every

        # Buffer: (tag, key) -> { "sum_value": ..., "sum_counter": ..., "count": ... }
        self._buffer = defaultdict(lambda: {"sum_value": 0.0,
                                            "sum_counter": 0.0,
                                            "count": 0})

    def __call__(self, data_dict, tag, counter, force=False):
        """
        Parameters
        ----------
        data_dict : dict
            {metric_name: value}
        tag : str
            Base tag for TensorBoard (e.g. "agents/run_0").
        counter : int or float
            X-axis value (step) for this log *if* it gets written.
        force : bool
            If True, write this point immediately to TensorBoard without
            touching the averaging buffer or global_counter.
        """
        if force:
            # Log this call immediately, bypassing averaging & global_counter
            for key, value in data_dict.items():
                self.writer.add_scalar(f"{tag}/{key}", value, counter)
            return

        # Normal averaged logging: count this call
        self.global_counter += 1

        # Accumulate into buffers
        for key, value in data_dict.items():
            buf = self._buffer[(tag, key)]
            buf["sum_value"] += float(value)
            buf["sum_counter"] += float(counter)
            buf["count"] += 1

        # Flush every N calls (across all tags/keys)
        if self.flush_every > 0 and (self.global_counter % self.flush_every == 0):
            self._flush_buffer()

    def _flush_buffer(self):
        """Write averaged metrics from buffer to TensorBoard, then clear the buffer."""
        for (tag, key), buf in self._buffer.items():
            if buf["count"] == 0:
                continue
            avg_value = buf["sum_value"] / buf["count"]
            avg_counter = buf["sum_counter"] / buf["count"]
            self.writer.add_scalar(f"{tag}/{key}", avg_value, avg_counter)

        # Clear buffer after flushing
        self._buffer.clear()

    def reset(self):
        """Reset global counter and internal buffers (but keep the same SummaryWriter)."""
        self.global_counter = 0
        self._buffer.clear()

    def close(self):
        """Flush any remaining buffered data and close the writer."""
        # Flush leftovers that didn't hit an exact multiple of flush_every
        if self._buffer:
            self._flush_buffer()
        self.writer.close()
        
class BasicCallBack:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_counter = 0

    def __call__(self, data_dict, tag, counter, force=None):
        self.global_counter += 1

        for key in data_dict:
            self.writer.add_scalar(f"{tag}/{key}", data_dict[key], counter)
    
    def reset(self):
        self.global_counter = 0
    
    def close(self):
        self.writer.close()
        
class EmptyCallBack:
    def __init__(self, log_dir):
        pass
    
    def __call__(self, data_dict, tag, counter, force=None):
        pass
    
    def reset(self):
        pass
    
    def close(self):
        pass        
        