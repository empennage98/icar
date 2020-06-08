import math
import numpy as np
import collections
import itertools

VERBOSE = False
BASELINE_MODE = False  # Switch between CaR as published in paper vs with improved constants.
if BASELINE_MODE:
  PHASE_1_CUTOFF_FACTOR = 1.5
else:
  PHASE_1_CUTOFF_FACTOR = 2

PartialExecution = collections.namedtuple('PartialExecution',
                                          'instance_id cumulative_runtime')


class CapsAndRuns(object):

  def __init__(self, env, set_t, get_t, config_ids, epsilon, delta, zeta, total_n, runtimeest_mode, baseline_mode, required_to_quantile_estimate=None):
    # runtimeest_mode can be:
    #   full: do a full phase I and phase II
    #   partial: pause phase II after b measurements
    #   skip: don't run phase II at all; in this case need to specify 
    #         required_to_quantile_estimate and only this many configurations
    #         from phase I will be finished
    assert runtimeest_mode in ['full', 'partial', 'skip']

    self.runtimeest_mode = runtimeest_mode
    self.env = env
    self.reject_ct_p1 = 0
    self.reject_ct_p2 = 0
    self.accept_ct = 0
    self.set_t = set_t
    self.get_t = get_t
    self.required_to_quantile_estimate = required_to_quantile_estimate
    self.accepted_phase1_skipped_phase2 = []
    self.config_ids = config_ids
    self.epsilon = epsilon
    self.delta = delta
    self.zeta = zeta
    self.n = total_n
    #if BASELINE_MODE:
    if baseline_mode:
      self.b = int(math.ceil(48 * math.log(3 * self.n / zeta) / delta))
      self.phase_1_cutoff_factor=2
    else:
      self.b = int(math.ceil(26 * math.log(2 * self.n / zeta) / delta))
      self.phase_1_cutoff_factor=1.5

    print('b: {}, n: {} len(configs): {}'.format(self.b, self.n, len(config_ids)))
    self.config_id_to_threads = {
        config_id: ExecutionThread(self, self.b, config_id,
                                   delta, epsilon, zeta, self.n, self.phase_1_cutoff_factor)
        for config_id in config_ids
    }
    self.best_avg, self.best_ucb, self.best_config = float('inf'), float('inf'), -1
    

  def run(self, stop_if_only_one_running):
    best_lb_for_finished_ones = None
    working_threads_left = len(self.config_id_to_threads)
    old_working_threads_left = working_threads_left
    original_working_threads_left = working_threads_left
    while working_threads_left > 0:
      if working_threads_left < original_working_threads_left and old_working_threads_left == original_working_threads_left:
        #if old_working_threads_left < float('inf'):
        print('phase 1 work until first complete: ' + format_runtime(
            np.sum([t.phase_1_work for t in self.config_id_to_threads.values()])) + ' global_t=' + str(self.get_t()))
        #print(working_threads_left)
        old_working_threads_left = working_threads_left
      if VERBOSE:
        print(working_threads_left)
      working_threads_left = 0
      for i, thread in enumerate(self.config_id_to_threads.values()):
        if thread.is_in_phase_2 or thread.is_in_phase_1:
          thread.step()
          if VERBOSE and thread.config_id == 943:
            print(
                'a={}, r1={}, r2={}, stepping {} phase1={}, phase2={}, next_instance_id={} its runtime={} tau={}, global_t={}'
                .format(self.accept_ct, self.reject_ct_p1, self.reject_ct_p2,
                        thread.config_id, thread.is_in_phase_1,
                        thread.is_in_phase_2,
                        thread.pending_measurements[0].instance_id,
                        thread.pending_measurements[0].cumulative_runtime,
                        thread.tau, self.get_t()))
          working_threads_left += 1
          the_one_working_thread = thread
        else:
          if self.best_avg > thread.result:
            self.best_avg = thread.result
            self.best_config = thread.config_id
          if self.best_ucb > thread.upper_bound:
            self.best_ucb = thread.upper_bound

      if self.runtimeest_mode == 'skip':
        assert self.required_to_quantile_estimate is not None
        if len(self.accepted_phase1_skipped_phase2) >= self.required_to_quantile_estimate:
          # We have enough finished, abort everything else.
          assert len(self.accepted_phase1_skipped_phase2) <= 2 * self.required_to_quantile_estimate, 'too many accepted at once, decrease resolution'
          # config_taus = {}
          # for i, thread in self.config_id_to_threads.items():
          #   config_taus[i] = np.max(thread.completed_runtimes)

          print('skip mode total work:',
          format_runtime(
              np.sum([
                  t.phase_1_work + t.phase_2_work
                  for t in self.config_id_to_threads.values()
              ])))

          return self.accepted_phase1_skipped_phase2[:self.required_to_quantile_estimate]

      if stop_if_only_one_running and working_threads_left == 1:
        if best_lb_for_finished_ones is None:
          best_lb_for_finished_ones = np.min([t.lower_bound for t in self.config_id_to_threads.values() 
                                              if t != the_one_working_thread])
        if best_lb_for_finished_ones > self.get_t(): 
          # Can reject everyone else in hindsight and only one remains:
          self.best_config = the_one_working_thread.config_id
          break
      else:
        best_lb_for_finished_ones = None

    #for t in self.config_id_to_threads.values():
    #  print(t.timeout, t.upper_bound, t.lower_bound)

    if self.runtimeest_mode == 'skip':
      # Done all of capsandruns but couldn't find the required number of good configs
      #print(len(self.accepted_phase1_skipped_phase2))
      assert len(self.accepted_phase1_skipped_phase2) <= self.required_to_quantile_estimate
      #print(len(self.config_id_to_threads.values()))
      print('WARNING: only found {} out of the required {} good configs, as most have been rejected'.format(
            len(self.accepted_phase1_skipped_phase2), self.required_to_quantile_estimate))
      return self.accepted_phase1_skipped_phase2

    print('phase 1 runtime min and max:',
        format_runtime(
            np.min([t.phase_1_work for t in self.config_id_to_threads.values()])),
        format_runtime(
            np.max([t.phase_1_work for t in self.config_id_to_threads.values()])))
    print('total phase 1+2 runtime:',
        format_runtime(
            np.sum([
                t.phase_1_work + t.phase_2_work
                for t in self.config_id_to_threads.values()
            ])))

    paused_threads_ct = 0
    for t in self.config_id_to_threads.values():
      if t.is_paused_phase2:
        assert self.runtimeest_mode != 'full'
        paused_threads_ct += 1
    if paused_threads_ct > 0:
      print('# of threads on pause: ' + str(paused_threads_ct))
    else:
      print(self.best_config, len(self.config_id_to_threads))
      if self.best_config == -1:
        return None

      return self.best_config, self.best_avg, self.best_ucb, self.config_id_to_threads[
          self.best_config].tau, np.sum(
              [t.phase_1_work for t in self.config_id_to_threads.values()])


class CapsAndRunsPool(CapsAndRuns):
  """Used to pool together a bunch of CapsAndRuns instances and resume all their paused threads."""
  def __init__(self, env, set_t, get_t, configs, pool, epsilon, delta, zeta, total_n, runtimeest_mode, baseline_mode):
    super(CapsAndRunsPool, self).__init__(env, set_t, get_t, [], epsilon, delta, zeta, total_n, runtimeest_mode, baseline_mode)
    assert runtimeest_mode == 'full'
    
    self.config_id_to_threads = {}
    num_active_threads = 0
    for p in pool:
      assert p.env == env
      assert p.set_t == set_t
      assert p.get_t == get_t
      assert p.epsilon == epsilon
      assert p.delta == delta
      assert p.n == total_n
      assert p.b == self.b
      assert p.zeta == zeta
      # do not require zeta to be the same
      self.reject_ct_p1 += p.reject_ct_p1
      self.reject_ct_p2 += p.reject_ct_p2
      self.accept_ct += p.accept_ct

      for config_id, thread in p.config_id_to_threads.items():
        assert config_id == thread.config_id
        assert config_id not in self.config_id_to_threads
        if config_id in configs:
          thread.car = self
          if thread.is_paused_phase2:
            thread.is_paused_phase2 = False
            thread.is_in_phase_2 = True
          if thread.is_in_phase_1 or thread.is_in_phase_2:  # either because just un-paused or was last one remaining in a previous car
            num_active_threads += 1
          self.config_id_to_threads[config_id] = thread

    print('don\'t believe len(configs) above')
    print('num active threads at resume: ' + str(num_active_threads))


class ExecutionThread(object):
  """Represents a simulated thread for measuring a particular configuration."""

  def __init__(self, car, b, config_id, delta, epsilon,
               zeta, n, phase_1_cutoff_factor):
    assert isinstance(config_id, int), config_id
    self.car = car
    self.b = b
    self.config_id = config_id
    self.delta = delta
    self.zeta = zeta
    self.epsilon = epsilon
    self.n = n
    self.phase_1_work = 0
    self.phase_2_work = 0
    self.sumq, self.sumq2 = 0, 0
    self.beta, self.kk = 1.10, 0
    self.work_step = 100  # todo
    self.how_many_to_complete = int(math.ceil(b * (1. - self.delta * 3 / 4)))
    self.is_in_phase_1, self.is_in_phase_2 = True, False
    self.is_paused_phase2 = False
    self.pending_measurements = [PartialExecution(j, 0) for j in range(b)]
    self.completed_runtimes = []
    self.result = None
    self.upper_bound = float('inf')
    self.lower_bound = 0.
    self.tau = None

    self.phase_1_cutoff_factor = phase_1_cutoff_factor

  def step(self):
    #print(self.config_id, self.phase_1_work, self.get_t())
    if self.is_in_phase_1:
      next_to_run = self.pending_measurements.pop(0)
      _, elapsed = self.car.env.run(
          self.config_id, next_to_run.instance_id,
          self.work_step + next_to_run.cumulative_runtime)
      #print(next_to_run.cumulative_runtime)
      assert elapsed >= next_to_run.cumulative_runtime
      self.phase_1_work += elapsed - next_to_run.cumulative_runtime
      #if self.phase_1_work >= PHASE_1_CUTOFF_FACTOR * self.b * self.car.get_t():
      if self.phase_1_work >= self.phase_1_cutoff_factor * self.b * self.car.get_t():
        # done, reject
        #print('reject {} {} {}'.format(self.config_id, self.phase_1_work, self.car.get_t()))
        self.car.reject_ct_p1 += 1
        self.result = float('inf')
        self.is_in_phase_1 = False
      else:
        if elapsed < self.work_step + next_to_run.cumulative_runtime:
          # this run is done
          self.completed_runtimes.append(elapsed)
          if len(self.completed_runtimes) >= self.how_many_to_complete:
            #print(self.completed_runtimes, self.completed_runtimes[-1], np.max(self.completed_runtimes))
            #print('done phase1 {} {}'.format(self.config_id, self.car.get_t()))
            assert self.completed_runtimes[-1] >= np.max(
                self.completed_runtimes) - self.work_step
            self.tau = np.max(self.completed_runtimes)
            self.completed_runtimes = []
            self.pending_measurements = [PartialExecution(self.b, 0)]
            self.is_in_phase_1 = False
            self.is_in_phase_2 = True
            self.car.accepted_phase1_skipped_phase2.append(self.config_id)
        else:
          # not done yet, add to the back of the queue
          last_to_run = PartialExecution(
              next_to_run.instance_id,
              next_to_run.cumulative_runtime + self.work_step)
          self.pending_measurements.append(last_to_run)

    elif self.is_in_phase_2:
      next_to_run = self.pending_measurements.pop(0)
      limit = min(self.work_step + next_to_run.cumulative_runtime, self.tau)
      #print(type(self.config_id), type(next_to_run.instance_id), type(limit))
      _, elapsed = self.car.env.run(self.config_id, next_to_run.instance_id,
                                    limit)
      assert elapsed >= next_to_run.cumulative_runtime
      self.phase_2_work += elapsed - next_to_run.cumulative_runtime
      if elapsed < limit or elapsed >= self.tau:
        # done with experiment
        self.completed_runtimes.append(elapsed)
        self.sumq += elapsed
        self.sumq2 += elapsed * elapsed
        self.pending_measurements = [
            PartialExecution(next_to_run.instance_id + 1, 0)
        ]

        # check for stopping condition
        j = next_to_run.instance_id - self.b
        q_mean = self.sumq / (j + 1)
        q_var = max((self.sumq2 - q_mean * self.sumq) / (j + 1), 0)

        if j + 1 > np.floor(
            np.power(self.beta,
                     self.kk)):  # source: "Empirical Bernstein Stopping" paper
          self.kk += 1
          alpha = (
              np.floor(np.power(self.beta, self.kk)) /
              np.floor(np.power(self.beta, self.kk - 1)))
          dk = self.zeta / (self.kk * (self.kk + 1) * self.n)
          dk = self.zeta / ((self.kk**1.1) * 10.5844 * self.n)
          self.x = -alpha * np.log(dk / 3)

        if j > 0:
          confidence = np.sqrt(q_var * 2 * self.x /
                               (j + 1)) + 3 * self.tau * self.x / (
                                   j + 1)
          self.lower_bound = q_mean - confidence
          self.upper_bound = q_mean + confidence
          self.car.set_t(min(self.car.get_t(), self.upper_bound), self.config_id)
          #print(upper_bound, self.car.get_t())

          if j == self.b:
            #assert confidence <= q_mean, confidence, 2 * q_mean
            if confidence > q_mean:
              print('!!!!!!!!!!!!! {} {}'.format(confidence, q_mean))
              self.car.set_t(min(self.car.get_t(), 2 * q_mean), self.config_id)
            if self.car.runtimeest_mode == 'partial':
              # done setting T up to constant accuracy
              self.result = float('inf')
              self.is_in_phase_2 = False
              self.is_paused_phase2 = True
          # if confidence <= q_mean and self.partial_runtimeest:
          #   # todo either document or remove
          #   # done setting T up to constant accuracy
          #   self.result = float('inf')
          #   self.is_in_phase_2 = False

          if self.lower_bound > self.car.get_t():
            # done, reject
            self.car.reject_ct_p2 += 1
            self.result = float('inf')
            self.is_in_phase_2 = False
          if confidence <= self.epsilon / 3 * (q_mean + self.lower_bound):  # used to be /3
          #if confidence <= self.epsilon / 5 * (q_mean + lower_bound):  # used to be /3
          #if confidence <= self.epsilon * 3 / 8 * q_mean:
          #if confidence <= self.epsilon / (2 + 2 * self.epsilon) * q_mean:
            #if confidence <= self.epsilon / ((2+self.epsilon)*(2+self.epsilon/(2+self.epsilon))) * (q_mean + lower_bound):
            # done, accept
            self.car.accept_ct += 1
            self.result = float(q_mean)
            self.is_in_phase_2 = False

      else:
        # not done yet
        last_to_run = PartialExecution(
            next_to_run.instance_id,
            next_to_run.cumulative_runtime + self.work_step)
        self.pending_measurements.append(last_to_run)


def format_runtime(runtime):
  return '{}s = {}m = {}h = {}d'.format(runtime, runtime / 60, runtime / 3600,
                                        runtime / (3600 * 24))
