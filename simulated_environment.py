import collections
import numpy as np
import os
import pickle

import json
from epm.surrogates.surrogate_model import SurrogateModel
from epm.webserver.flask_server_helper import handle_request

TOTAL_INSTANCES = 50000  # 168000
# GENERATED_DIR = 'generated'  # generated_overtenthofasec
# PID = os.getpid()

def get_random_settings(cs):
    param = cs.sample_configuration().get_dictionary()
    param_arg = []
    for key in param.keys():
        param_arg.append('-' + str(key))
        param_arg.append(str(param[key]))

    return param_arg

class RegionEnvironment():
    def __init__(self, seed=520): 
        model_dir = '/home/wlin/car/cplex_regions200_trim/'

        self.model = SurrogateModel(
            pyrfr_wrapper=os.path.join(model_dir, 'pyrfr_wrapper.out.par10.pkl'),
            pyrfr_model=os.path.join(model_dir, 'pyrfr_model.out.par10.bin'),
            config_space=os.path.join(model_dir, 'config_space.out.par10.pcs'),
            inst_feat_dict=os.path.join(model_dir, 'inst_feat_dict.out.par10.json'),
            idle_time=60,
            impute_with='default',
            quality=False,
            dtype=np.float32,
            debug=False,
        )

        self.surrogate_model, self.inst_feat_dict, self.cs = self.model.load_model()
        self.cs.seed(seed)

        self.cached_runtimes = collections.defaultdict(dict)
        self.run_so_far = collections.defaultdict(dict)
        self.random_configs = [get_random_settings(self.cs)]
        self.total_runtime = 0

        self.counter = 0

        self.rs = np.random.RandomState(seed)
        self.instance_perm = self.rs.permutation(TOTAL_INSTANCES)

    def _run(self, config_id, instance_id, timeout):

        # Check Cache
        instance_id = self.instance_perm[instance_id % TOTAL_INSTANCES] + 1
        if instance_id in self.cached_runtimes[config_id]:
            return min(self.cached_runtimes[config_id][instance_id], timeout)

        self.counter += 1
        if self.counter % 100000 == 0:
            print(self.counter, self.total_runtime)

        data = {}
        data['instance_name'] = str(instance_id) + '.lp'
        data['instance_info'] = 0
        data['cutoff'] = 2147483647
        data['runlength'] = 2147483647
        data['seed'] = 1
        data['params'] = self.random_configs[config_id]

        json_data = json.loads(json.dumps(data))

        rt = min(10000, handle_request(json_data=json_data, surrogate_model=self.model)[0][0])

        self.cached_runtimes[config_id][instance_id] = rt

        return min(rt, timeout)

    def run(self, config_id, instance_id, timeout):

        if config_id >= len(self.random_configs):
            # Generate new random configuration.
            self.random_configs.append(get_random_settings(self.cs))

        # Resue instance if we run out of them
        # instance_id = instance_id % TOTAL_INSTANCES + 1

        #if instance_id in self.cached_runtimes[config_id]:
        #    run_so_far, runlimit = self.cached_runtimes[config_id][instance_id]
        #    if run_so_far < runlimit or run_so_far >= timeout:
        #        # Re-use runtime
        #        return run_so_far < timeout, min(run_so_far, timeout)

        #runlimit = 0.1
        ##runlimit = 10
        #while runlimit < timeout:
        #    runlimit *= 2

        # Query EPM
        runtime = self._run(config_id, instance_id, timeout)
        if instance_id in self.run_so_far[config_id]:
            if runtime > self.run_so_far[config_id][instance_id]:
                self.total_runtime += runtime - self.run_so_far[config_id][instance_id]
                self.run_so_far[config_id][instance_id] = runtime
        else:
            self.total_runtime += runtime
            self.run_so_far[config_id][instance_id] = runtime

        # Caching/bookkeeping.
        #if instance_id in self.cached_runtimes[config_id]:
        #    run_so_far, _ = self.cached_runtimes[config_id][instance_id]
        #    if run_so_far >= min(runtime, timeout) + 1e-6:
        #        print('ran for less than should: {}, {}, {}'.format(
        #            run_so_far, runtime, timeout))
        #        runtime = run_so_far
        #    self.total_runtime -= run_so_far

        #self.total_runtime += min(runtime, timeout)
        #self.cached_runtimes[config_id][instance_id] = (runtime, timeout)
        # print('ran config_id={}, instance_id={}, timeout={}, finished={}'.format(
        #     config_id, instance_id, timeout, runtime < timeout))

        return runtime < timeout, min(runtime, timeout)

class ExpEnvironment():
    def __init__(self, seed=520): 

        self.rs = np.random.RandomState(seed)

        self.random_configs = np.load('mean_50000.npy')[:10000]
        self.cached_runtimes = []
        for i, m in enumerate(self.random_configs):
            self.cached_runtimes.append(self.rs.exponential(m, TOTAL_INSTANCES))
        self.cached_runtimes = np.array(self.cached_runtimes)
        self.run_so_far = collections.defaultdict(dict)

        self.total_runtime = 0
        self.counter = 0

    def _run(self, config_id, instance_id, timeout):

        # Check Cache
        instance_id = instance_id % TOTAL_INSTANCES
        config_id = config_id % 10000
        #if instance_id in self.cached_runtimes[config_id]:
        #    return min(self.cached_runtimes[config_id][instance_id], timeout)

        #self.counter += 1
        #if self.counter % 1000000 == 0:
        #    print(self.counter, self.total_runtime)

        rt = self.cached_runtimes[config_id,instance_id]

        return min(rt, timeout)

    def run(self, config_id, instance_id, timeout):

        #if config_id >= len(self.random_configs):
        #    # Generate new random configuration.
        #    self.random_configs.append(get_random_settings(self.cs))

        # Query
        runtime = self._run(config_id, instance_id, timeout)
        if instance_id in self.run_so_far[config_id]:
            if runtime > self.run_so_far[config_id][instance_id]:
                self.total_runtime += runtime - self.run_so_far[config_id][instance_id]
                self.run_so_far[config_id][instance_id] = runtime
        else:
            self.total_runtime += runtime
            self.run_so_far[config_id][instance_id] = runtime

        return runtime < timeout, min(runtime, timeout)
