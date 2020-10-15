import matplotlib.pyplot as plt
import pickle
import numpy as np

################
# Table 1. CPU Runtime & #. of Conf.
def runtime(rt_log, gammas=[0.05,0.02,0.01]):
    gamma_seq = [0.05, 0.02, 0.01]
    time_icar, time_icar_std = [], []
    time_car_v2, time_car_v2_std = [], []
    time_car_v1, time_car_v1_std = [], []
    num_before = []
    num_after, num_after_std = [], []
    num_car_v2 = []
    num_car_v1 = []
    
    for gamma in gammas:
        # ICAR
        rt_list = []
        num_after_list = []
        num_before_list = []
        for log in rt_log:
            if log[0]['gamma'] == gamma and log[0]['only_car'] == False and log[0]['baseline_mode'] == False and log[0]['guessbestconf'] == False:
                num_after_list.append(log[1])
                num_before_list.append(log[2])
                rt_list.append(log[3])
        rt_list = np.array(rt_list) / 86400
        assert len(rt_list) == 5
        assert len(num_after_list) ==5
        assert len(num_before_list) == 5
        time_icar.append(np.mean(rt_list))
        time_icar_std.append(np.std(rt_list, ddof=1))
        num_before.append(np.mean(num_before_list))
        num_after.append(np.mean(num_after_list))
        num_after_std.append(np.std(num_after_list, ddof=1))

        # CAR++
        rt_list = []
        num_car_list = []
        for log in rt_log:
            if log[0]['gamma'] == gamma and log[0]['only_car'] == True and log[0]['baseline_mode'] == False:
                num_car_list.append(log[1])
                rt_list.append(log[3])
        rt_list = np.array(rt_list) / 86400
        assert len(rt_list) == 5
        time_car_v2.append(np.mean(rt_list))
        time_car_v2_std.append(np.std(np.array(rt_list), ddof=1))
        num_car_v2.append(np.mean(num_car_list))
        
        # CAR
        rt_list = []
        num_car_list = []
        for log in rt_log:
            if log[0]['gamma'] == gamma and log[0]['only_car'] == True and log[0]['baseline_mode'] == True:
                num_car_list.append(log[1])
                rt_list.append(log[3])
        rt_list = np.array(rt_list) / 86400
        assert len(rt_list) == 5
        assert len(num_car_list) == 5
        time_car_v1.append(np.mean(rt_list))
        time_car_v1_std.append(np.std(np.array(rt_list), ddof=1))
        num_car_v1.append(np.mean(num_car_list))
    
    print('CPU Time')
    print('ICAR', np.round(time_icar), np.round(time_icar_std))
    print('CAR++', np.round(time_car_v2), np.round(time_car_v2_std))
    print('CAR', np.round(time_car_v1), np.round(time_car_v1_std))

    print('# of Conf.')
    print('ICAR', num_before, np.round(num_after))
    print('CAR++', num_car_v2)
    print('CAR', num_car_v1)
    
print('Sat')
with open('./sat/sat_a_rt_delta0.1.dump', 'rb') as f:
    rt_car_log = pickle.load(f)
    runtime(rt_car_log)
print('Region')
with open('./region/region_a_rt_delta0.1.dump', 'rb') as f:
    rt_car_log = pickle.load(f)
    runtime(rt_car_log)
print('RCW')
with open('./reproduce/rcw/rcw_a_rt_delta0.1.dump', 'rb') as f:
    rt_car_log = pickle.load(f)
    runtime(rt_car_log)

    
################   
# Table. 1 R^delta of returned conf.
def Rdelta(rt, rt_log, delta=0.1):
    rt_delta = []
    for i in range(5):
        noi = rt[i].shape[1]
        rt_trim = np.sort(rt[i], axis=-1)[:,:int(noi*(1-delta))]
        rt_delta.append(np.mean(rt_trim, axis=-1))
    
    icar, icar_std = [], []
    car_v2, car_v2_std = [], []
    car_v1, car_v1_std = [], []
    for gamma in [0.05, 0.02, 0.01]:
        icar_rt, car_v2_rt, car_v1_rt = [], [], []
        for log in rt_car_log:
            if log[0]['gamma'] == gamma and log[0]['only_car'] == True and log[0]['baseline_mode'] == False:
                car_v2_rt.append(rt_delta[(log[0]['seed'] - 520)][log[-1][0]])
            if log[0]['gamma'] == gamma and log[0]['only_car'] == True and log[0]['baseline_mode'] == True:
                car_v1_rt.append(rt_delta[(log[0]['seed'] - 520)][log[-1][0]])
            if log[0]['gamma'] == gamma and log[0]['only_car'] == False and log[0]['baseline_mode'] == False and log[0]['guessbestconf'] == False:
                icar_rt.append(rt_delta[(log[0]['seed'] - 520)][log[-1][0]])
        icar_rt = np.array(icar_rt)
        car_v2_rt = np.array(car_v2_rt)
        car_v1_rt = np.array(car_v1_rt)
        assert len(icar_rt) == 5
        assert len(car_v2_rt) == 5
        assert len(car_v1_rt) == 5
        icar.append(np.mean(icar_rt))
        icar_std.append(np.std(icar_rt, ddof=1))
        car_v2.append(np.mean(car_v2_rt))
        car_v2_std.append(np.std(icar_rt, ddof=1))
        car_v1.append(np.mean(car_v1_rt))
        car_v1_std.append(np.std(car_v1_rt, ddof=1))
    print('ICAR', ['{:.1f}'.format(i) for i in icar], ['{:.1f}'.format(i) for i in icar_std])
    print('CAR++', ['{:.1f}'.format(i) for i in car_v2], ['{:.1f}'.format(i) for i in car_v2_std])
    print('CAR', ['{:.1f}'.format(i) for i in car_v1], ['{:.1f}'.format(i) for i in car_v1_std])
    
# SAT
rt = []
with open('./dataset/minisat_cnfuzzdd/measurements.dump', 'rb') as f:
    rt_data = pickle.load(f)
    rt_data = [rt_data[k] for k in sorted(rt_data.keys())]
for i in range(5):
    rs = np.random.RandomState(seed=520+i)
    instance_perm = rs.permutation(len(rt_data[0]))
    config_perm = rs.permutation(len(rt_data))
    rt_i = np.stack([np.array(rt_data[config_perm[k]]) for k in range(len(rt_data))])
    rt_i[rt_i >= 900] = 900
    rt.append(rt_i)
print('SAT')
with open('./sat/sat_a_rt_delta0.1.dump', 'rb') as f:
    rt_car_log = pickle.load(f)
    Rdelta(rt, rt_car_log)
    
# Region
rt = []
for i in range(5):
    rt.append(np.load('./dataset/cplex_region/cplex_region_rt_seed{}.npy'.format(520+i)))
print('Region')
with open('./region/region_a_rt_delta0.1.dump', 'rb') as f:
    rt_car_log = pickle.load(f)
    Rdelta(rt, rt_car_log)
    
# RCW
rt = []
for i in range(5):
    rt.append(np.load('./dataset/cplex_rcw/cplex_rcw_rt_seed{}.npy'.format(520+i)))
print('RCW')
with open('./rcw/rcw_a_rt_delta0.1.dump', 'rb') as f:
    rt_car_log = pickle.load(f)
    Rdelta(rt, rt_car_log)
    
################
# Figure 1
def preprocess(rt, rdelta):
    total_rt = -np.array([sum(rt[i].values()) for i in rt]) / 86400
    idx = np.argsort(rdelta[:total_rt.shape[0]])

    return -total_rt[idx]

def preprocess_sat(rt, rt_data):
    rs = np.random.RandomState(seed=520)
    instance_perm = rs.permutation(len(rt_data[0]))
    config_perm = rs.permutation(len(rt_data))
    noi = len(rt_data[0])

    rt_data = np.array([rt_data[config_perm[i]] for i in range(rt.shape[0])])
    rt_data[rt_data > 900.] = 900.
    rt_sort_and_trim = np.sort(rt_data, axis=-1)[:,:int(noi*(1-0.1))]
    rdelta = np.mean(rt_sort_and_trim, axis=-1)
    
    total_rt = -np.array(rt) / 86400
    idx = np.argsort(rdelta[:total_rt.shape[0]])

    return -total_rt[idx]

def time_per_conf(rt, rsf, title, fn):
    noi = rt.shape[1]
    rt_sort_and_trim = np.sort(rt, axis=-1)[:,:int(noi*(1-0.1))]
    r_delta = np.mean(rt_sort_and_trim, axis=-1)

    plt.figure(figsize=(4,3))
    rt_sum = preprocess(rsf['icar'], r_delta)
    plt.plot(rt_sum[:], label='ICAR')
    rt_sum = preprocess(rsf['car_v2'], r_delta)
    plt.plot(rt_sum[:], label='CAR++')
    rt_sum = preprocess(rsf['car_v1'], r_delta)
    plt.plot(rt_sum[:], label='CAR')

    plt.legend(frameon=False)
    plt.xlabel('Configurations (sorted with $\delta$-capped mean)')
    plt.ylabel('CPU time (days)')
    plt.yscale('log')
    #plt.ylim(0.2,40)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fn, dpi=300)
    
def time_per_conf_sat(rt_data, rsf, title, fn):
    plt.figure(figsize=(4,3))
    rt_sum = preprocess_sat(np.array(list(rsf['icar'].values())), rt_data)
    plt.plot(rt_sum[:], label='ICAR')
    rt_sum = preprocess_sat(np.array(list(rsf['car_v2'].values())), rt_data)
    plt.plot(rt_sum[:], label='CAR++')
    rt_sum = preprocess_sat(np.array(list(rsf['car_v1'].values())), rt_data)
    plt.plot(rt_sum[:], label='CAR')

    plt.legend(frameon=False)
    plt.xlabel('Configurations (sorted with $\delta$-capped mean)')
    plt.ylabel('CPU time (days)')
    plt.yscale('log')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fn, dpi=300)
    
# SAT
rsf = {}
with open('./sat/sat_a_rsf_epsilon0.05_delta0.1_gamma0.05_carFalse_baselineFalse_gbcFalse_seed520.dump', 'rb') as f:
    rsf['icar'] = pickle.load(f)
with open('./sat/sat_a_rsf_epsilon0.05_delta0.1_gamma0.05_carTrue_baselineFalse_gbcTrue_seed520.dump', 'rb') as f:
    rsf['car_v2'] = pickle.load(f)
with open('./sat/sat_a_rsf_epsilon0.05_delta0.1_gamma0.05_carTrue_baselineTrue_gbcTrue_seed520.dump', 'rb') as f:
    rsf['car_v1'] = pickle.load(f)
with open('./dataset/minisat_cnfuzzdd/measurements.dump', 'rb') as f:
    rt_data = pickle.load(f)
    rt_data = [rt_data[k] for k in sorted(rt_data.keys())]
time_per_conf_sat(rt_data, rsf, 'Minisat/CNFuzzDD', 'sat_work_on_conf.png')

# Region
rsf = {}
with open('./region/region_a_rsf_epsilon0.05_delta0.1_gamma0.05_carFalse_baselineFalse_gbcFalse_seed520.dump', 'rb') as f:
    rsf['icar'] = pickle.load(f)
with open('./region/region_a_rsf_epsilon0.05_delta0.1_gamma0.05_carTrue_baselineFalse_gbcTrue_seed520.dump', 'rb') as f:
    rsf['car_v2'] = pickle.load(f)
with open('./region/region_a_rsf_epsilon0.05_delta0.1_gamma0.05_carTrue_baselineTrue_gbcTrue_seed520.dump', 'rb') as f:
    rsf['car_v1'] = pickle.load(f)
rt = np.load('./dataset/cplex_region/cplex_region_rt_seed520.npy')
time_per_conf(rt, rsf, 'CPLEX/Regions200', 'region_work_on_conf.png')

# RCW
rsf = {}
with open('./rcw/rcw_a_rsf_epsilon0.05_delta0.1_gamma0.05_carFalse_baselineFalse_gbcFalse_seed520.dump', 'rb') as f:
    rsf['icar'] = pickle.load(f)
with open('./rcw/rcw_a_rsf_epsilon0.05_delta0.1_gamma0.05_carTrue_baselineFalse_gbcTrue_seed520.dump', 'rb') as f:
    rsf['car_v2'] = pickle.load(f)
with open('./rcw/rcw_a_rsf_epsilon0.05_delta0.1_gamma0.05_carTrue_baselineTrue_gbcTrue_seed520.dump', 'rb') as f:
    rsf['car_v1'] = pickle.load(f)
rt = np.load('./dataset/cplex_rcw/cplex_rcw_rt_seed520.npy')
time_per_conf(rt, rsf, 'CPLEX/RCW', 'rcw_work_on_conf.png')


################
# Figure 2. Distribution of R^delta
rt_all = {}
# SAT
with open('./dataset/minisat_cnfuzzdd/measurements.dump', 'rb') as f:
    rt_data = pickle.load(f)
    rt = np.stack([np.array(rt_data[k]) for k in sorted(rt_data.keys())])
    rt[rt >= 900] = 900
rt_all['Minisat/CNFuzzDD'] = rt

# Region
rt = []
for i in range(5):
    rt.append(np.load('./dataset/cplex_region/cplex_region_rt_seed{}.npy'.format(520+i)))
rt = np.concatenate(rt)
rt_all['CPLEX/Regions200'] = rt

# RCW
rt = []
for i in range(5):
    rt.append(np.load('./dataset/cplex_rcw/cplex_rcw_rt_seed{}.npy'.format(520+i)))
rt = np.concatenate(rt)
rt_all['CPLEX/RCW'] = rt

delta = 0.1
r_delta = {}
for dataset in rt_all:
    n = int(rt_all[dataset].shape[1]*(1-delta))
    rt_sort = np.sort(rt_all[dataset], axis=-1)
    r_delta[dataset] = np.mean(rt_sort[:,:n], axis=-1)

fn = ['sat', 'region', 'rcw']
xs = []
weights = []
labels = []
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for idx, data in enumerate(r_delta):
    cnt, bins = np.histogram(r_delta[data], bins=np.logspace(np.log10(5),np.log10(10000), 40))
    cnt = cnt / len(r_delta[data])
    xs.append(bins[:-1])
    weights.append(cnt)
    labels.append(data)
    plt.figure(figsize=(4,4))
    plt.hist(bins[:-1], bins, weights=cnt, label=data, color=colors[idx])
    plt.xscale('log')
    plt.title(data)
    plt.xlabel('$R^\delta$ of configurations')
    plt.ylabel('Proportion (%)')
    plt.tight_layout()
    plt.savefig('hist_{}.png'.format(fn[idx]), dpi=300)
    
################
# Table 2. CPU Runtime & #. of Conf.
for m in [2, 5, 10, 25]:
    print(m)
    with open('./uniform/rt_epsilon0.05_delta0.1_gamma0.02_m{}.dump'.format(m), 'rb') as f:
        rt_car_log = pickle.load(f)
        runtime(rt_car_log, gammas=[0.02])
        
################      
# Figure 3
rsf_car_v2 = {}
rsf_car_v1 = {}
rsf_icar = {}
for m in [2, 5, 10, 25]:
    with open('./uniform/rsf_epsilon0.05_delta0.1_gamma0.02_m{}_carTrue_baselineFalse_gbcTrue_seed520.dump'.format(m), 'rb') as f:
        rsf_car_v2[m] = pickle.load(f)
    with open('./uniform/rsf_epsilon0.05_delta0.1_gamma0.02_m{}_carTrue_baselineTrue_gbcTrue_seed520.dump'.format(m), 'rb') as f:
        rsf_car_v1[m] = pickle.load(f)
    with open('./uniform/rsf_epsilon0.05_delta0.1_gamma0.02_m{}_carFalse_baselineFalse_gbcFalse_seed520.dump'.format(m), 'rb') as f:
        rsf_icar[m] = pickle.load(f)

for m in [2, 5, 10, 25]:
    rs = np.random.RandomState(seed=520)
    r_delta = rs.uniform(10, 10 * m, size=1000) # Use r instead since they are monotonic

    plt.figure(figsize=(3,3))

    rt_sum = preprocess(rsf_icar[m], r_delta)
    plt.plot(rt_sum[:], label='ICAR')
    rt_sum = preprocess(rsf_car_v2[m], r_delta)
    plt.plot(rt_sum[:], label='CAR++')
    rt_sum = preprocess(rsf_car_v1[m], r_delta)
    plt.plot(rt_sum[:], label='CAR')

    plt.legend(frameon=False, fontsize='small', loc=1)
    plt.xlabel('Configurations \n (sorted with $\delta$-capped mean)')
    plt.ylabel('CPU time (days)')
    plt.yscale('log')
    plt.ylim(0.05, 10)
    plt.title('c={}'.format(m))
    plt.tight_layout()
    plt.savefig('c{}_work_on_conf.png'.format(m), dpi=300)

################
# Table 3
def vary_params(rt_log):
    table = []
    for delta in [0.025, 0.05, 0.075, 0.1]:
        row = []
        for epsilon in [0.025, 0.05, 0.075, 0.1]:
            rt_icar = []
            rt_car = []
            for log in rt_log:
                if log[0]['delta'] == delta and log[0]['epsilon'] == epsilon:
                    if log[0]['only_car'] == False and log[0]['baseline_mode'] == False and log[0]['guessbestconf'] == False:
                        rt_icar.append(log[3])
                    if log[0]['only_car'] == True and log[0]['baseline_mode'] == False:
                        rt_car.append(log[3])
            assert len(rt_icar) == 5
            assert len(rt_car) == 5
            rt_icar = np.mean(rt_icar)
            rt_car = np.mean(rt_car)
            print('epsillon {} delta {}: {:.2f}'.format(epsilon, delta, rt_car/rt_icar))
            row.append(rt_car / rt_icar)
        print('')
        table.append(row)

# SAT
print('SAT')
with open('./sat/sat_vary_rt.dump'.format(delta), 'rb') as f:
    rt_log = pickle.load(f)
    vary_params(rt_log)
    
# Region
print('Region')
with open('./region/region_vary_rt.dump'.format(delta), 'rb') as f:
    rt_log = pickle.load(f)
    vary_params(rt_log)

# RCW
print('RCW')
with open('./rcw/rcw_vary_rt.dump'.format(delta), 'rb') as f:
    rt_log = pickle.load(f)
    vary_params(rt_log)

################
# Table 3 ratio of R^delta of returned conf.
def Rdelta_vary(rt, rt_log):
    for delta in [0.025, 0.05, 0.075, 0.1]:
        # Calculate R^delta
        rt_delta = []
        for i in range(5):
            noi = rt[i].shape[1]
            rt_trim = np.sort(rt[i], axis=-1)[:,:int(noi*(1-delta))]
            rt_delta.append(np.mean(rt_trim, axis=-1))
        for epsilon in [0.025, 0.05, 0.075, 0.1]:
            rt_icar = []
            rt_car = []
            for log in rt_log:
                if log[0]['delta'] == delta and log[0]['epsilon'] == epsilon:
                    if log[0]['only_car'] == False and log[0]['baseline_mode'] == False and log[0]['guessbestconf'] == False:
                        rt_icar.append(rt_delta[(log[0]['seed'] - 520)][log[-1][0]])
                    if log[0]['only_car'] == True and log[0]['baseline_mode'] == False:
                        rt_car.append(rt_delta[(log[0]['seed'] - 520)][log[-1][0]])
            print('epsilon {}, delta {}, ratio {:.2f}'.format(epsilon, delta, np.mean(rt_icar) / np.mean(rt_car)))
            
# SAT
rt = []
with open('./dataset/minisat_cnfuzzdd/measurements.dump', 'rb') as f:
    rt_data = pickle.load(f)
    rt_data = [rt_data[k] for k in sorted(rt_data.keys())]
for i in range(5):
    rs = np.random.RandomState(seed=520+i)
    instance_perm = rs.permutation(len(rt_data[0]))
    config_perm = rs.permutation(len(rt_data))
    rt_i = np.stack([np.array(rt_data[config_perm[k]]) for k in range(len(rt_data))])
    rt_i[rt_i >= 900] = 900
    rt.append(rt_i)
print('SAT')
with open('./sat/sat_vary_rt.dump', 'rb') as f:
    rt_car_log = pickle.load(f)
    Rdelta_vary(rt, rt_car_log)
    
# Region
rt = []
for i in range(5):
    rt.append(np.load('./dataset/cplex_region/cplex_region_rt_seed{}.npy'.format(520+i)))
print('Region')
with open('./region/region_vary_rt.dump', 'rb') as f:
    rt_car_log = pickle.load(f)
    Rdelta_vary(rt, rt_car_log)
    
# RCW
rt = []
for i in range(5):
    rt.append(np.load('./dataset/cplex_rcw/cplex_rcw_rt_seed{}.npy'.format(520+i)))
print('RCW')
with open('./rcw/rcw_vary_rt.dump', 'rb') as f:
    rt_car_log = pickle.load(f)
    Rdelta_vary(rt, rt_car_log)