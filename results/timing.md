| dataset | e-means s/epoch | k-means s/iter | e-means epochs → time to reach converged k-means* | k-means iters → time | speedup | k-means@50 time |
|---|---|---|---|---|---|---|
| gmm-d32 | 0.58 | 0.50 | 1 → 0.6 s | 20 → 10.3 s | **16.5×** | 19.5 s |
| uniform-d64 | 0.63 | 0.54 | 1 → 0.6 s | 2 → 1.1 s | **1.9×** | 27.4 s |
| gmm-d2 | 0.06 | 0.01 | 3 → 0.3 s | 50 → 0.5 s | **2.1×** | 0.5 s |
| covtype | 1.37 | 1.56 | never | 50 → 66.0 s | — | 66.0 s |
| covtype (revival off) | 1.37 | 1.56 | 2 → 2.8 s | 50 → 66.0 s | **23.9×** | 66.0 s |
| covtype-minmax | 1.37 | 1.29 | 3 → 4.1 s | 50 → 64.4 s | **15.6×** | 64.4 s |
| mnist | 0.17 | 0.10 | 2 → 0.3 s | 30 → 2.8 s | **8.3×** | 5.1 s |
