from simulated_environment import RuntimeMatrixEnvironment
from impatient import ImpatientCapsAndRuns
import capsandruns as capsandruns

import numpy as np

import pickle

# Use WORK_STEP = 10

# only_car, baseline_mode, gbc
configs = [
    (False, False, False),
    (True, False, False),
]

all_total_runtimes = []

for seed in range(5):
    for epsilon in [0.025, 0.05, 0.075, 0.1]:
        for delta in [0.025, 0.05, 0.075, 0.1]:
            for gamma in [0.02]:
                # Adaptive K
                max_k = int(np.ceil(np.log(0.5/gamma) / np.log(2)))
                print(max_k)
                for only_car, baseline_mode, guessbestconf in configs:
                    settings = {
                        'epsilon': epsilon,
                        'delta': delta,
                        'gamma': gamma,
                        'max_k': max_k,
                        'zeta': 0.05,
                        'only_car': only_car,
                        'baseline_mode': baseline_mode,
                        'guessbestconf': guessbestconf,
                        'seed': 520 + seed
                    }
                    print(settings)
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

                    with open('region/region_vary_rsf_epsilon{}_delta{}_gamma{}_car{}_baseline{}_gbc{}_seed{}.dump'.format(settings['epsilon'],delta,gamma, only_car, baseline_mode, guessbestconf, 520+seed), 'wb') as f:
                        pickle.dump(icar.env.run_so_far, f)
                    with open('region/region_vary_rt.dump', 'wb') as f:
                        pickle.dump(all_total_runtimes, f)
print("Done")
