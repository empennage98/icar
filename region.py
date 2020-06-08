from simulated_environment import RuntimeMatrixEnvironment
from impatient import ImpatientCapsAndRuns
import capsandruns as capsandruns

import numpy as np

import pickle

# Use WORK_STEP = 10

# only_car, baseline_mode, gbc
configs = [
    (False, False, True),
    (False, False, False),
    (True, True, True),
    (True, False, True)
]

all_total_runtimes = []

delta = 0.1

for seed in range(5):
    for gamma in [0.05, 0.02, 0.01]:
        # Adaptive K
        max_k = int(np.ceil(np.log(0.5/gamma) / np.log(2)))
        print(max_k)
        for only_car, baseline_mode, guessbestconf in configs:
            print(gamma)
            settings = {
                'epsilon': 0.05,
                'delta': delta,
                'gamma': gamma,
                'max_k': max_k,
                'zeta': 0.05,
                'only_car': only_car,
                'baseline_mode': baseline_mode,
                'guessbestconf': guessbestconf,
                'seed': 520 + seed
            }
            env = RuntimeMatrixEnvironment('./dataset/cplex_region/cplex_region_rt_seed{}.npy'.format(520+seed))
            icar = ImpatientCapsAndRuns(env=env, **settings)
            ret = icar.run()
            all_total_runtimes.append((settings,
                icar.final_precheck_accepted, 
                icar.final_precheck_examined,
                icar.env.total_runtime, 
                capsandruns.format_runtime(icar.env.total_runtime),
                ret))
            print('global_t: {}'.format(icar.global_t))
            print(all_total_runtimes)

            with open('region/region_a_rsf_epsilon{}_delta{}_gamma{}_car{}_baseline{}_gbc{}_seed{}.dump'.format(settings['epsilon'],delta,gamma, only_car, baseline_mode, guessbestconf, 520+seed), 'wb') as f:
                pickle.dump(icar.env.run_so_far, f)
            with open('region/region_a_rt_delta{}.dump'.format(delta), 'wb') as f:
                pickle.dump(all_total_runtimes, f)
print("Done")
