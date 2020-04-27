import math
import numpy as np
import collections
import itertools

R = 48  #0.1  # 48
VERBOSE = False

PartialExecution = collections.namedtuple('PartialExecution',
                                          'instance_id cumulative_runtime')


class CapsAndRuns(object):

  def __init__(self, env, set_t, get_t):
    self.env = env
    self.reject_ct_p1 = 0
    self.reject_ct_p2 = 0
    self.accept_ct = 0
    self.set_t = set_t
    self.get_t = get_t

  def run(self, config_ids, epsilon, delta, zeta, partial_runtimeest):
    n = len(config_ids)
    b = int(math.ceil(R * math.log(3 * n / zeta) / delta))
    print('b: {}, n: {}'.format(b, n))
    config_id_to_threads = {
        config_id: ExecutionThread(self, partial_runtimeest, b, config_id,
                                   delta, epsilon, zeta, n)
        for config_id in config_ids
    }
    best_avg, best_ucb, best_config = float('inf'), float('inf'), -1
    working_threads_left = n
    old_working_threads_left = working_threads_left
    original_working_threads_left = working_threads_left
    while working_threads_left > 1:
      if working_threads_left < original_working_threads_left and old_working_threads_left == original_working_threads_left:
        #if old_working_threads_left < float('inf'):
        print('phase 1 work until first complete: ' + format_runtime(
            np.sum([t.phase_1_work for t in config_id_to_threads.values()])))
        #print(working_threads_left)
        old_working_threads_left = working_threads_left
      if VERBOSE:
        print(working_threads_left)
      working_threads_left = 0
      for i, thread in enumerate(config_id_to_threads.values()):
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
        else:
          if best_avg > thread.result:
            best_avg = thread.result
            best_config = thread.config_id
          if best_ucb > thread.upper_bound:
            best_ucb = thread.upper_bound

    if working_threads_left == 1:
      for i, thread in enumerate(config_id_to_threads.values()):
        if thread.is_in_phase_2 or thread.is_in_phase_1:
          best_config = thread.config_id
          # assert best_avg > thread.result
          # best_avg = thread.result
          # if best_ucb > thread.upper_bound:
          #   best_ucb = thread.upper_bound

    print('phase 1 runtime min and max:',
        format_runtime(
            np.min([t.phase_1_work for t in config_id_to_threads.values()])),
        format_runtime(
            np.max([t.phase_1_work for t in config_id_to_threads.values()])))
    print('total phase 1+2 runtime:',
        format_runtime(
            np.sum([
                t.phase_1_work + t.phase_2_work
                for t in config_id_to_threads.values()
            ])))
    print(best_config, len(config_id_to_threads))
    if best_config == -1:
      return None

    return best_config, best_avg, best_ucb, config_id_to_threads[
        best_config].tau, np.sum(
            [t.phase_1_work for t in config_id_to_threads.values()])


class ExecutionThread(object):
  """Represents a simulated thread for measuring a particular configuration."""

  def __init__(self, car, partial_runtimeest, b, config_id, delta, epsilon,
               zeta, n):
    self.car = car
    self.partial_runtimeest = partial_runtimeest
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
    self.work_step = 10  # todo
    self.how_many_to_complete = int(math.ceil(b * (1. - self.delta * 3 / 4)))
    self.is_in_phase_1, self.is_in_phase_2 = True, False
    self.pending_measurements = [PartialExecution(j, 0) for j in range(b)]
    self.completed_runtimes = []
    self.result = None
    self.upper_bound = float('inf')
    self.tau = None

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
      if self.phase_1_work >= 2 * self.b * self.car.get_t():
        # done, reject
        self.car.reject_ct_p1 += 1
        self.result = float('inf')
        self.is_in_phase_1 = False
      else:
        if elapsed < self.work_step + next_to_run.cumulative_runtime:
          # this run is done
          self.completed_runtimes.append(elapsed)
          if len(self.completed_runtimes) >= self.how_many_to_complete:
            #print(self.completed_runtimes, self.completed_runtimes[-1], np.max(self.completed_runtimes))
            assert self.completed_runtimes[-1] >= np.max(
                self.completed_runtimes) - self.work_step
            self.tau = np.max(self.completed_runtimes)
            self.completed_runtimes = []
            self.pending_measurements = [PartialExecution(self.b, 0)]
            self.is_in_phase_1 = False
            self.is_in_phase_2 = True
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
          lower_bound = q_mean - confidence
          upper_bound = q_mean + confidence
          self.upper_bound = upper_bound
          self.car.set_t(min(self.car.get_t(), upper_bound))
          #print(upper_bound, self.car.get_t())

          if j == self.b:
            #assert confidence <= q_mean, confidence, 2 * q_mean
            if confidence > q_mean:
              print('!!!!!!!!!!!!! {} {}'.format(confidence, q_mean))
              self.car.set_t(min(self.car.get_t(), 2 * q_mean))
            if self.partial_runtimeest:
              # done setting T up to constant accuracy
              self.result = float('inf')
              self.is_in_phase_2 = False
          # if confidence <= q_mean and self.partial_runtimeest:
          #   # todo either document or remove
          #   # done setting T up to constant accuracy
          #   self.result = float('inf')
          #   self.is_in_phase_2 = False

          if lower_bound > self.car.get_t():
            # done, reject
            self.car.reject_ct_p2 += 1
            self.result = float('inf')
            self.is_in_phase_2 = False
          #if confidence <= self.epsilon / 3 * (q_mean + lower_bound):  # used to be /3
          #if confidence <= self.epsilon / 5 * (q_mean + lower_bound):  # used to be /3
          #if confidence <= self.epsilon * 3 / 8 * q_mean:
          if confidence <= self.epsilon / (2 + 2 * self.epsilon) * q_mean:
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
