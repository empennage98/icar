import simulated_environment
import capsandruns

import numpy as np
from scipy import stats
import pickle
import sys

# Not so impatient version
class ImpatientCapsAndRuns(object):
  def __init__(self, epsilon, delta, gamma, r, p, max_k, zeta, n=None, only_car=False, haystack_gamma=None, seed=520):
    #self.env = simulated_environment.MinisatEnvironment(fake=True)
    #self.env = simulated_environment.PrerunEnvironment(haystack_gamma=haystack_gamma)
    self.env = simulated_environment.RegionEnvironment(seed=seed)
    #self.env = simulated_environment.ExpEnvironment()
    self.epsilon = epsilon
    self.delta = delta
    self.gamma = gamma
    self.r = r
    self.p = p
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

  def get_t(self):
    return self.global_t

  def set_t(self, t):
    assert t <= self.global_t
    self.global_t = t

  def run_car_only(self):
    # baseline
    zeta = self.zeta / 7
    n = self.n
    if self.n is None:
      n = int(np.ceil(np.log(zeta) / np.log(1-self.gamma)))
    self.final_precheck_accepted = n
    self.final_precheck_examined = n
    configs = list(range(n))
    car = capsandruns.CapsAndRuns(self.env, self.set_t, self.get_t)
    car_results = car.run(configs, self.epsilon, self.delta, zeta, False)
    print(car_results)
    print(capsandruns.format_runtime(self.env.total_runtime))
    
  def precheck(self, failure_probab, gamma, do_any_reject=True):
    print(failure_probab, gamma)
    if do_any_reject:
      assert self.global_t < float('inf')
    # todo tune these constants
    first_precheck_runs = 80
    second_precheck_runs = 100
    first_precheck_delta_cap = 0.1
    # Total failure probability is 2*failure_probab, distributed equally between the
    # two failure cases in Lemma 2 and 4.
    # Working with preliminary calculations that ensure good config rejected wp <=2e^-8.
    if do_any_reject:
      num_optimal_configs_needed = np.ceil(np.log(failure_probab) / np.log(2*np.exp(-8))) 
    else:
      num_optimal_configs_needed = 1
    assert num_optimal_configs_needed >= 1

    # Calculate num_configs_to_examine
    # Config is optimal wp gamma.
    # binary search for the sample size that ensures 
    # num_optimal_configs_needed optimal configs.
    # Use inverse cdf of the binomial for this.
    ub = 1.  # Upper bound on num_configs_to_examine.
    while stats.binom.ppf(failure_probab, ub, gamma) < num_optimal_configs_needed:
      ub *= 2
    lb = ub / 2
    while ub - lb > 0.5:
      if stats.binom.ppf(failure_probab, (lb+ub)/2, gamma) < num_optimal_configs_needed:
        lb = (lb+ub)/2
      else:
        ub = (lb+ub)/2
    num_configs_to_examine = int(np.ceil(ub))

    precheck_configs = list(range(num_configs_to_examine))
    if not do_any_reject:
      return precheck_configs

    passed_configs = []
    for i in precheck_configs:
      np.random.seed(i)
      # First phase: measure caps, because we have to ensure we cap at least delta/2 fraction.
      # To do this quickly we use a constant capping (1/10 now) so few samples give high
      # probability bound.
      runtimes = []
      for _ in range(first_precheck_runs):
        j = np.random.randint(50000)
        runtimes.append(self.env.run(i, j, 5 * self.global_t)[1])  # todo tune constant 5
      runtimes = sorted(runtimes, reverse=True)
      cap = runtimes[int(np.ceil(first_precheck_runs*first_precheck_delta_cap))]

      # Second phase, average measurement. We use new measurements because they need to be iid.
      runtimes = []
      for _ in range(second_precheck_runs):
        j = np.random.randint(50000)
        runtimes.append(self.env.run(i, j, cap)[1])
      avg_runtime = np.mean(runtimes)
      if avg_runtime < 2 * self.global_t:
        passed_configs.append(i)

    print('precheck accepted {} fraction of configs out of {}'.format(
              float(len(passed_configs))/len(precheck_configs), len(precheck_configs)))
    self.final_precheck_accepted = len(passed_configs)
    self.final_precheck_examined = len(precheck_configs)
        
    return passed_configs

  def run(self):
    zeta = self.zeta / 25
    for k in range(self.max_k - 1, -2, -1):
      print('k=' + str(k))
      if k == -1:
        configs = self.precheck(6*zeta / self.max_k, self.gamma)
        car = capsandruns.CapsAndRuns(self.env, self.set_t, self.get_t)
        car_results = car.run(configs, self.epsilon, self.delta, zeta, False)
        print(car_results)
        print(capsandruns.format_runtime(self.env.total_runtime))
        return car_results
      else:#231.44970675476353d
        gamma = 2 ** k * self.gamma
        configs = self.precheck(6 * zeta / self.max_k, gamma, 
          do_any_reject=k < self.max_k - 1)
        car = capsandruns.CapsAndRuns(self.env, self.set_t, self.get_t)
        car.run(configs, self.epsilon, self.delta/2, zeta/self.max_k, True)
      print('current runtime {}'.format(self.env.total_runtime))

if __name__ == '__main__':
  all_total_runtimes = []

  #max_k = 2
  #delta = 0.1

  #p = 10
  #r = 2
  #gamma = 0.05
  #print(gamma, r,p)
  #settings = {
  #  'epsilon': 0.05,
  #  'delta': delta,
  #  'gamma': gamma,
  #  'r': r,
  #  'p': p,
  #  'max_k': max_k,
  #  'zeta': 0.05,
  #  'haystack_gamma': gamma,
  #  'only_car': False,
  #}
  #icar = ImpatientCapsAndRuns(**settings)
  #icar.run()
  #all_total_runtimes.append((settings, icar.final_precheck_accepted, 
  #  icar.final_precheck_examined, icar.env.total_runtime, 
  #  capsandruns.format_runtime(icar.env.total_runtime)))
  #print('global_t: {}'.format(icar.global_t))
  #exit(0)

  max_k = 5
  # for gamma_i in range(1):  # 20
    # gamma = 0.003 + gamma_i * 0.0025  # 0.0035
  #gamma = float(sys.argv[1])
  delta = 0.1
  #print(delta, gamma)

  r = 2
  p = 2

  #for r in [2]: # Rejection multiplier
  #  for p in [2]: # Enlarging multiplier
  for seed in range(5):
    for gamma in [0.05, 0.02, 0.01]:
      print(gamma, r,p)
      settings = {
        'epsilon': 0.05,
        'delta': delta,
        'gamma': gamma,
        'r': r,
        'p': p,
        'max_k': max_k,
        'zeta': 0.05,
        'haystack_gamma': gamma,
        'only_car': False,
        'seed': 520 + seed
      }
      icar = ImpatientCapsAndRuns(**settings)
      ret = icar.run()
      all_total_runtimes.append((settings, icar.final_precheck_accepted, 
        icar.final_precheck_examined, icar.env.total_runtime, 
        capsandruns.format_runtime(icar.env.total_runtime),
        ret))
      print('global_t: {}'.format(icar.global_t))
      exit(0)

      print(all_total_runtimes)
      with open('reg_icar_three_epsilon{}_delta{}_gamma{}_r{}_p{}_seed{}.dump'.format(settings['epsilon'],delta,gamma,r,p,520+seed), 'wb') as f:
        pickle.dump(all_total_runtimes, f)
      with open('reg_icar_three_rsf_epsilon{}_delta{}_gamma{}_r{}_p{}_seed{}.dump'.format(settings['epsilon'],delta,gamma,r,p,520+seed), 'wb') as f:
        pickle.dump(icar.env.run_so_far, f)
      #with open('icar_newzeta_rt_epsilon{}_delta{}_gamma{}_r{}_p{}.dump'.format(settings['epsilon'],delta,gamma,r,p), 'wb') as f:
      #  pickle.dump(icar.env.cached_runtimes, f)
  

  #print('----------------')
  #settings = {
  #  'epsilon': 0.2,
  #  'delta': 0.1,
  #  'gamma': gamma,
  #  'max_k': max_k,
  #  'zeta': 0.05,
  #  'haystack_gamma': gamma,
  #  'only_car': True,
  #}
  #icar = ImpatientCapsAndRuns(**settings)
  ##icar = ImpatientCapsAndRuns(0.2, 0.05, None, 7, 0.5, n=9000, only_car=True)
  #icar.run()
  #all_total_runtimes.append((settings, icar.final_precheck_accepted, 
  #  icar.final_precheck_examined, icar.env.total_runtime, 
  #  capsandruns.format_runtime(icar.env.total_runtime)))
  #print('global_t: {}'.format(icar.global_t))

  # for i in range(1, 20):
  #   gamma = 0.0035
  #   settings = {
  #     'epsilon': 0.2,
  #     'delta': 0.1,
  #     'gamma': gamma,
  #     'max_k': i,
  #     'zeta': 0.05,
  #     'haystack_gamma': gamma,
  #   }
  #   icar = ImpatientCapsAndRuns(**settings)
  #   #icar = ImpatientCapsAndRuns(0.2, 0.05, None, 7, 0.5, n=9000)
  #   icar.run()
  #   all_total_runtimes.append((settings, icar.final_precheck_accepted, 
  #     icar.final_precheck_examined, icar.env.total_runtime, 
  #     capsandruns.format_runtime(icar.env.total_runtime)))
