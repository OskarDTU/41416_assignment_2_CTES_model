from src.features.ctes_1d_jian import init_ctes_state, step_ctes, extract_profiles
import numpy as np, time

y = init_ctes_state(T_init=130.0)
m = 0.021 * 870.0
Tin = 242.0
bad = False
t0=time.time()
for i in range(1, 61):
    r = step_ctes(y, Tin, m, 'charging', 600)
    y = r['y']
    Tf, Ts = extract_profiles(y)
    mx = max(float(np.max(Tf)), float(np.max(Ts)))
    mn = min(float(np.min(Tf)), float(np.min(Ts)))
    if i % 10 == 0:
        print('step',i,'T_out',r['T_out_C'],'Tfmax',float(np.max(Tf)),'Tsmax',float(np.max(Ts)),'elapsed',time.time()-t0)
    if (not np.isfinite(mx)) or mx > 1000 or mn < -200:
        print('BAD at', i, 'T_out', r['T_out_C'], 'mx', mx, 'mn', mn, 'energy', r['energy_J'], 'close', r.get('diag_closure_combined_W'))
        bad = True
        break
if not bad:
    print('OK final', 'T_out', r['T_out_C'], 'Tfmax', float(np.max(Tf)), 'Tsmax', float(np.max(Ts)), 'energy', r['energy_J'],'elapsed',time.time()-t0)
