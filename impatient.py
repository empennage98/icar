# Code taken from https://github.com/empennage98/icar

import simulated_environment
import capsandruns as capsandruns

import math
import numpy as np
from scipy import stats
import pickle
import sys

WORK_STEP = 100
PHASE_1_CUTOFF_FACTOR = 1.9  # for precheck
PHASE_2_CUTOFF_FACTOR = 2.99  # for precheck

class ImpatientCapsAndRuns(object):
  def __init__(self, epsilon, delta, gamma, max_k, zeta, n=None, only_car=False, haystack_gamma=None, seed=520, env=None, baseline_mode=True, guessbestconf=False):
    #self.env = simulated_environment.MinisatEnvironment(fake=True)
    #self.env = simulated_environment.PrerunEnvironment(haystack_gamma=haystack_gamma)
    assert haystack_gamma is None
    #self.env = simulated_environment.WeibullEnvironment(seed=seed)
    if env is not None:
        self.env = env
    self.epsilon = epsilon
    self.delta = delta
    self.gamma = gamma
    self.max_k = max_k
    self.zeta = zeta
    self.global_t = float('inf')
    self.n = n
    if n is not None:
      assert gamma is None
      self.gamma = 5 * np.log(self.max_k / (self.zeta / 25 * 6)) / n
      # assert np.ceil(
      #     8*np.log(self.max_k / (self.zeta/15)) / self.gamma) == n, (8*np.log(self.max_k / (self.zeta/15)) / self.gamma, n)
    if only_car:
      self.run = self.run_car_only
    self.final_precheck_accepted = None
    self.final_precheck_examined = None
    self.total_precheck_time = 0
    self.config_last_set_t = None

    self.baseline_mode = baseline_mode
    self.guessbestconf = guessbestconf

  def get_t(self):
    return self.global_t

  def set_t(self, t, i):
    assert t <= self.global_t
    self.global_t = t
    self.config_last_set_t = i

  def run_car_only(self):
    # baseline
    zeta = self.zeta / 7
    n = self.n
    if self.n is None:
      n = int(np.ceil(np.log(zeta) / np.log(1-self.gamma)))
    self.final_precheck_accepted = n
    self.final_precheck_examined = n
    configs = list(range(n))
     # env, set_t, get_t, config_ids, epsilon, delta, zeta, total_n
    car = capsandruns.CapsAndRuns(self.env, self.set_t, self.get_t, configs, self.epsilon, 
      self.delta, zeta, len(configs), 'full', self.baseline_mode)
    car_results = car.run(stop_if_only_one_running=True)
    print(car_results)
    print(capsandruns.format_runtime(self.env.total_runtime))

    return car_results

  def calc_precheck_pool_sizes(self, failure_probab, gamma):
    # Total failure probability is failure_probab
    
    self.precheck_pool_ranges = {}
    for k in range(self.max_k - 1, -1, -1):
      gamma = self.gamma * 2 ** k
      range_to = int(np.ceil(np.log(failure_probab / self.max_k) / np.log(1 - gamma)))
      if k < self.max_k - 1:
        range_from = self.precheck_pool_ranges[k+1][1]
      else:
        range_from = 0
      self.precheck_pool_ranges[k] = (range_from, range_to)
    self.precheck_pool_ranges[-1] = (0, int(np.ceil(np.log(failure_probab / self.max_k) / np.log(1 - self.gamma))))
    assert self.precheck_pool_ranges[-1][1] == self.precheck_pool_ranges[0][1]

  def precheck(self, configs, failure_probab):
    # Total failure probability is 3*failure_probab
    assert self.global_t < float('inf')
    precheck_start_time = self.env.total_runtime
    # todo tune these constants
    b_prime = int(np.ceil(32.1 * np.log(2*self.max_k/failure_probab)))
    print('b_prime={}'.format(b_prime))
    
    passed_configs = []
    for i in configs:
      if i == self.config_last_set_t:
        passed_configs.append(i)
      else:
        # First phase: measure caps, because we have to ensure we cap at least delta/2 fraction.
        # To do this quickly we use a constant capping (2/10 now) so few samples give high
        # probability bound.
        pending_measurements = [capsandruns.PartialExecution(j, 0) for j in range(b_prime)]

        # Phase I
        phase_1_work = 0
        rejected = False
        completed_runtimes = []
        how_many_to_complete = int(math.ceil(b_prime * (0.8)))
        while True:
          next_to_run = pending_measurements.pop(0)
          _, elapsed = self.env.run(
              i, next_to_run.instance_id, WORK_STEP + next_to_run.cumulative_runtime)
          #print(next_to_run.cumulative_runtime)
          assert elapsed >= next_to_run.cumulative_runtime
          phase_1_work += elapsed - next_to_run.cumulative_runtime
          #print(phase_1_work, len(completed_runtimes), how_many_to_complete, b_prime, self.global_t)
          if phase_1_work >= PHASE_1_CUTOFF_FACTOR * b_prime * self.global_t:
            # done, reject
            rejected = True
            #print('reject {} cap={}, avg runtime={} agains {}'.format(i, elapsed, phase_1_work/b_prime, PHASE_1_CUTOFF_FACTOR * self.global_t))
            break
          else:
            if elapsed < WORK_STEP + next_to_run.cumulative_runtime:
              # this run is done
              completed_runtimes.append(elapsed)
              if len(completed_runtimes) >= how_many_to_complete:
                #print('done phase1 {} {}'.format(self.config_id, self.car.get_t()))
                assert completed_runtimes[-1] >= np.max(completed_runtimes) - WORK_STEP
                tau = np.max(completed_runtimes)
                break
            else:
              # not done yet, add to the back of the queue
              last_to_run = capsandruns.PartialExecution(
                  next_to_run.instance_id,
                  next_to_run.cumulative_runtime + WORK_STEP)
              pending_measurements.append(last_to_run)

        # Phase II
        phase_2_work = 0
        if not rejected:
          # Second phase, average measurement. We use new measurements because they need to be iid.
          runtimes = []
          for j in range(b_prime, 2*b_prime):
            t = self.env.run(i, j, tau)[1]
            phase_2_work += t
            runtimes.append(t)
            if phase_2_work > PHASE_2_CUTOFF_FACTOR * b_prime * self.global_t:
              break
          avg_runtime = np.mean(runtimes)
          # empirical Bernstein confidence
          confidence = (np.std(runtimes) * np.sqrt(2 * np.log(3 * self.max_k / failure_probab) / len(runtimes)) + 
                        3 * tau * np.log(3 * self.max_k / failure_probab) / len(runtimes))
          if avg_runtime - confidence <= self.global_t:
            passed_configs.append(i)
            assert isinstance(i, int)

    for i in passed_configs:
      assert isinstance(i, int)

    print('precheck accepted {} fraction of configs out of {}'.format(
              float(len(passed_configs))/len(configs), len(configs)))
    self.final_precheck_accepted = len(passed_configs)
    self.final_precheck_examined = len(configs)
    precheck_end_time = self.env.total_runtime
    self.total_precheck_time += precheck_end_time - precheck_start_time
        
    return passed_configs


  def run(self):
    # so far: 6 from capsandruns, 1 from guessbestconfigs, 1 from sampling good configs, 3 from precheck
    # old: 2 from lemma 4, 1 from lemma 5 and 6 from capsandruns, 2 from precheck, 1 from very initial car
    _passed_precheck = []
    if self.guessbestconf:
      zeta = self.zeta / 13  # todo needs updating
    else:
      zeta = self.zeta / 12
    self.calc_precheck_pool_sizes(zeta, self.gamma)
    print('precheck pool ranges: {}'.format(self.precheck_pool_ranges))
    assert self.precheck_pool_ranges[-1][0] == 0
    assert self.precheck_pool_ranges[self.max_k-1][0] == 0
    all_configs = list(range(self.precheck_pool_ranges[-1][0], self.precheck_pool_ranges[-1][1]))

    # env, set_t, get_t, pool, epsilon, delta, zeta, total_n
    # env, set_t, get_t, config_ids, epsilon, delta, zeta, total_n
    if self.guessbestconf:
      car = capsandruns.CapsAndRuns(self.env, self.set_t, self.get_t, all_configs, 1, 0.75, zeta/6, len(all_configs), 'skip', self.baseline_mode, required_to_quantile_estimate=self.precheck_pool_ranges[self.max_k-1][1])
      good_configs = car.run(stop_if_only_one_running=True)
    else:
      good_configs = []
    print(good_configs)
    for i in good_configs:
      assert isinstance(i, int)
    assert len(good_configs) <= self.precheck_pool_ranges[self.max_k-1][1]

    # Reset global_t as we're now switching to the real delta.
    self.global_t = float('inf')
    pool = []
    configs_dealt_with_already = set()

    for k in range(self.max_k - 1, -2, -1):
      print('k=' + str(k))
      if k == -1:
        configs = list(range(self.precheck_pool_ranges[-1][0], self.precheck_pool_ranges[-1][1]))
        print(configs)
        configs = self.precheck(configs, zeta)
        print(configs)
        assert len(configs) > 0
        car = capsandruns.CapsAndRunsPool(self.env, self.set_t, self.get_t, configs, pool, self.epsilon, self.delta, 
          zeta, len(all_configs), 'full', self.baseline_mode)
        car_results = car.run(stop_if_only_one_running=True)
        print(car_results)
        print(capsandruns.format_runtime(self.env.total_runtime))
      else:
        gamma = 2 ** k * self.gamma
        print('gamma:', gamma)
        assert gamma < 1
        configs = list(range(self.precheck_pool_ranges[k][0], self.precheck_pool_ranges[k][1]))
        print(configs)
        if k == self.max_k - 1:
          # First iteration.
          configs.extend(good_configs)
          configs = list(set(configs))

        configs = [c for c in configs if c not in configs_dealt_with_already]
        configs_dealt_with_already.update(configs)
        if k < self.max_k - 1:
          configs = self.precheck(configs, zeta)
        _passed_precheck.extend(configs)
        print(configs)
        if len(configs) == 0:
          print('all configs eliminated in this iteration, skipping to next')
        else:
          car = capsandruns.CapsAndRuns(self.env, self.set_t, self.get_t, configs, self.epsilon, self.delta, zeta, 
            len(all_configs), 'partial', self.baseline_mode)
          car.run(stop_if_only_one_running=False)
          pool.append(car)

      print('current runtime {}'.format(self.env.total_runtime))
    print('total precheck time {}'.format(capsandruns.format_runtime(self.total_precheck_time)))

    return car_results
