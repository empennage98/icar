import collections
import pickle
import numpy as np

class Environment(object):
    """This class is used for simulating runs and collecting statistics."""

    def __init__(self, results_file, timeout, seed=520):
        """Prepares an instance that can simulate runs based on a measurements file.

        Args:
          results_file: the location of the pickle dump containing the results
            of the runtime measurements.
          timeout: the timeout used for the runtime measurements.
        """
        self._timeout = timeout

        # Load measurements.
        with open(results_file, 'rb') as f:
            results = pickle.load(f)

        # random.seed(1234) # random shuffle of instance order (for all configs)
        # n_instances = len(results.values()[0])
        # shuffle_mask = range(n_instances)
        # random.shuffle(shuffle_mask)
        # self._results = [list(np.array(results[k])[shuffle_mask]) for k in sorted(results.keys())]

        self._results = [results[k] for k in sorted(results.keys())]


        self.rs = np.random.RandomState(seed)

        self._instance_count = len(self._results[0])
        self.instance_perm = self.rs.permutation(self._instance_count)
        self._config_count = len(results)
        self.config_perm = self.rs.permutation(self._config_count)

        print(self._instance_count)
        print(self.instance_perm[:10])
        print(self._config_count)
        print(self.config_perm[:10])

        self.reset()

    def reset(self):
        """Reset the state of the environment."""
        # Total runtime, without resuming, of any configuration ran on any instance.
        # Without resuming means that if the same configuration-instance pair is
        # run, `total_runtime` will track the time taken as if the execution had to
        # be restarted, rather than resumed from when it was last interrupted
        # due to a timeout.
        self.total_runtime = 0
        # Total runtime, with resuming, of any configuration ran on any instance.
        self._total_resumed_runtime = 0
        # Dict mapping a configuration to how long it was run, with resuming, on all
        # instances combined.
        self._runtime_per_config = collections.defaultdict(float)
        # Dict mapping a configuration and an instance to how long it ran so far
        # in total, with resuming. Summing the runtimes for all instances for a
        # configuration will be equal to the relevant value in `runtime_per_config`.
        self.run_so_far = collections.defaultdict(lambda: collections.defaultdict(float))
        #self.run_so_far = collections.defaultdict(dict)

    def get_num_configs(self):
        return len(self._results)

    def get_num_instances(self):
        return self._instance_count

    def gettotal_runtime(self):
        return self.total_runtime

    def get_total_resumed_runtime(self):
        return self._total_resumed_runtime

    def get_results(self):
        return self._results

    def get_runtime_per_config(self):
        return self._runtime_per_config

    def run(self, config_id, instance_id, timeout):
        """Simulates a run of a configuration on an instance with a timeout.

        Args:
          config_id: specifies which configuration to run. Integer from 0 to
            get_num_configs() - 1.
          instance_id: the instance to run. If not specified, a random instance
            will be run.
          timeout: the timeout to simulate the run with.

        Raises:
          ValueError: if the supplied timeout is larger than self.timeout, the
            requested simulation cannot be completed and this error will be raised.

        Returns:
          A tuple of whether the simulated run timed out, and how long it ran.
        """
        if timeout > self._timeout:
            raise ValueError('timeout provided is too high to be simulated. timeout={}'.format(timeout))
        if instance_id is None:
            instance_id = np.random.randint(self._instance_count)

        origin_instance_id = self.instance_perm[instance_id % self._instance_count]
        origin_config_id = self.config_perm[config_id % self._config_count]

        #runtime = min(timeout, self._results[origin_config_id][instance_id % self._instance_count])
        runtime = min(timeout, self._results[origin_config_id][origin_instance_id])

        resumed_runtime = runtime - self.run_so_far[config_id][instance_id]
        #self.total_runtime += runtime
        # Consider environment supporting resuming
        self.total_runtime += resumed_runtime

        self._runtime_per_config[config_id] += resumed_runtime
        self.run_so_far[config_id][instance_id] = runtime
        self._total_resumed_runtime += resumed_runtime

        return timeout <= self._results[origin_config_id][origin_instance_id], runtime

    def print_config_stats(self, config_id, tau=None):
        """Prints statistics about a particular configuration."""

        # Compute average runtime capped at TIMEOUT.
        average = np.mean([min(self._timeout, r) for r in self._results[config_id]])
        print('avg runtime capped at the dataset\'s timeout: {}'.format(average))
        timeout_count = 0
        for t in self._results[config_id]:
            if t > self._timeout:
                timeout_count += 1
        print('fraction of instances timing out at the timeout of the dataset: {}'.format(float(timeout_count) / len(self._results[config_id])))
        if tau is not None:
            timeout_count = 0
            for t in self._results[config_id]:
                if t > tau:
                    timeout_count += 1
            print('fraction of instances timing out at tau: {}'.format(float(timeout_count) / len(self._results[config_id])))
        # Disable log
        # with open('runtime_per_config.dump', 'wb') as outf:
        #     pickle.dump(self._runtime_per_config, outf)
