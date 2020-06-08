import collections
import numpy as np
import os
import pickle

import json

class RuntimeMatrixEnvironment():
    def __init__(self, fn, seed=520):

        # fn is the filename of the runtime matrix
        self.rt = np.load(fn)
        self.run_so_far = collections.defaultdict(dict)
        self.total_runtime = 0

        self.rs = np.random.RandomState(seed)
        self.total_instances = self.rt.shape[1]
        self.instance_perm = self.rs.permutation(self.total_instances)

    def _run(self, config_id, instance_id, timeout):

        instance_id = self.instance_perm[instance_id % self.total_instances]

        return min(self.rt[config_id,instance_id], timeout)

    def run(self, config_id, instance_id, timeout):

        runtime = self._run(config_id, instance_id, timeout)
        if instance_id in self.run_so_far[config_id]:
            if runtime > self.run_so_far[config_id][instance_id]:
                self.total_runtime += runtime - self.run_so_far[config_id][instance_id]
                self.run_so_far[config_id][instance_id] = runtime
        else:
            self.total_runtime += runtime
            self.run_so_far[config_id][instance_id] = runtime

        return runtime < timeout, min(runtime, timeout)

class ExpEnvironment():
    def __init__(self, seed, max_multiplier): 

        self.rs = np.random.RandomState(seed)

        self.random_configs = self.rs.uniform(10, 10 * max_multiplier, size=1000)

        self.total_instances = 50000
        self.cached_runtimes = []
        for i, m in enumerate(self.random_configs):
            self.cached_runtimes.append(self.rs.exponential(m, self.total_instances))
        self.cached_runtimes = np.array(self.cached_runtimes)
        self.run_so_far = collections.defaultdict(dict)

        self.total_runtime = 0
        self.counter = 0

    def _run(self, config_id, instance_id, timeout):

        # Check Cache
        instance_id = instance_id % TOTAL_INSTANCES
        config_id = config_id % 1000

        rt = self.cached_runtimes[config_id,instance_id]

        return min(rt, timeout)

    def run(self, config_id, instance_id, timeout):

        runtime = self._run(config_id, instance_id, timeout)
        if instance_id in self.run_so_far[config_id]:
            if runtime > self.run_so_far[config_id][instance_id]:
                self.total_runtime += runtime - self.run_so_far[config_id][instance_id]
                self.run_so_far[config_id][instance_id] = runtime
        else:
            self.total_runtime += runtime
            self.run_so_far[config_id][instance_id] = runtime

        return runtime < timeout, min(runtime, timeout)
