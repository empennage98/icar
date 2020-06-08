from simulated_environment import ExpEnvironment
from impatient import ImpatientCapsAndRuns
import capsandruns as capsandruns

import numpy as np

import pickle

# Use workstep = 10

# only_car, baseline_mode, gbc
configs = [
    (False, False, True),
    (False, False, False),
    (True, True, True),
    (True, False, True)
]

if __name__ == '__main__':

    max_k = 5
    delta = 0.1

    gamma = 0.02

    for max_multiplier in [2, 5, 10, 25]:
        all_total_runtimes = []
        for seed in range(5):
            for only_car, baseline_mode, guessbestconf in configs:
                max_k = int(np.ceil(np.log(0.5/gamma) / np.log(2)))
                print(max_multiplier, max_k, gamma)
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
                env = ExpEnvironment(520+seed, max_multiplier)
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
                print(ret)

                with open('uniform/rsf_epsilon{}_delta{}_gamma{}_m{}_car{}_baseline{}_gbc{}_seed{}.dump'.format(settings['epsilon'],delta,gamma,max_multiplier,only_car, baseline_mode, guessbestconf, 520+seed), 'wb') as f:
                    pickle.dump(icar.env.run_so_far, f)

                with open('uniform/rt_epsilon{}_delta{}_gamma{}_m{}.dump'.format(settings['epsilon'],delta,gamma,max_multiplier), 'wb') as f:
                    pickle.dump(all_total_runtimes, f)
print("Done writing")
