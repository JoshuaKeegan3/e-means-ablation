## Datasets

| dataset | n train | d | k |
|---|---|---|---|
| gmm-d32 | 200000 | 32 | 1024 |
| uniform-d64 | 200000 | 64 | 1024 |
| gmm-d2 | 200000 | 2 | 256 |
| covtype | 500000 | 54 | 1024 |
| covtype-minmax | 500000 | 54 | 1024 |
| mnist | 60000 | 784 | 256 |

## Ladder: strip e-means down to Lloyd one ingredient at a time

Held-out MSE, mean paired % difference over seeds. `vs km@T`: against faiss k-means run for the same number of epochs (negative = e-means variant better). `vs km@50`: against k-means at 50 iterations (≈ converged).

### T = 1 epochs

| rung | gmm-d32 vs km@T | vs km@50 | uniform-d64 vs km@T | vs km@50 | gmm-d2 vs km@T | vs km@50 | covtype vs km@T | vs km@50 | covtype-minmax vs km@T | vs km@50 | mnist vs km@T | vs km@50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 e-means | -32.1% | +0.1% | -0.2% | +0.0% | -28.6% | +11.7% | -47.6% | +120.7% | -29.6% | +2.5% | -4.7% | +0.9% |
| L1 - revival | -26.3% | +8.6% | +1.4% | +1.7% | -28.6% | +11.7% | -48.5% | +117.6% | -29.8% | +2.3% | -3.9% | +1.7% |
| L2 - closing means | -27.0% | +7.6% | +1.5% | +1.8% | -31.2% | +7.6% | -67.7% | +46.9% | -30.5% | +1.3% | -4.0% | +1.5% |
| L3 - cosine schedule | -27.1% | +7.5% | +1.8% | +2.1% | -33.9% | +3.3% | -77.8% | -5.3% | -30.3% | +1.5% | -2.0% | +3.7% |
| L4 - EMA (= Sculley mini-batch) | -24.9% | +10.8% | +1.1% | +1.3% | -22.2% | +21.6% | -44.2% | +130.3% | -27.8% | +5.1% | -3.5% | +2.1% |
| L5 - mini-batch (= Lloyd) | -0.0% | +47.4% | -0.0% | +0.2% | +0.0% | +56.5% | +0.0% | +328.9% | +0.0% | +45.9% | +0.0% | +5.8% |

 Step impact (Δ vs previous rung, % of held-out MSE; positive = removing it hurt):

| removed | gmm-d32 | uniform-d64 | gmm-d2 | covtype | covtype-minmax | mnist |
|---|---|---|---|---|---|---|
| revival | +8.5% | +1.7% | +0.0% | -1.5% | -0.2% | +0.8% |
| closing means | -0.9% | +0.1% | -3.6% | -35.9% | -1.0% | -0.1% |
| cosine schedule | -0.1% | +0.3% | -4.0% | -24.9% | +0.3% | +2.1% |
| EMA (= Sculley mini-batch) | +3.0% | -0.8% | +17.7% | +153.8% | +3.5% | -1.5% |
| mini-batch (= Lloyd) | +33.1% | -1.0% | +28.6% | +83.0% | +38.9% | +3.7% |

### T = 2 epochs

| rung | gmm-d32 vs km@T | vs km@50 | uniform-d64 vs km@T | vs km@50 | gmm-d2 vs km@T | vs km@50 | covtype vs km@T | vs km@50 | covtype-minmax vs km@T | vs km@50 | mnist vs km@T | vs km@50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 e-means | -19.7% | -5.5% | +0.0% | +0.0% | -27.2% | +1.9% | +101.1% | +127.1% | -14.1% | +0.3% | -2.7% | +0.1% |
| L1 - revival | -13.5% | +1.8% | +1.0% | +1.0% | -27.2% | +1.9% | -21.6% | -11.3% | -14.6% | -0.3% | -2.3% | +0.5% |
| L2 - closing means | -13.5% | +1.8% | +1.0% | +1.0% | -27.4% | +1.6% | -21.6% | -11.4% | -14.6% | -0.3% | -2.3% | +0.5% |
| L3 - cosine schedule | -13.1% | +2.3% | +1.3% | +1.3% | -28.8% | -0.4% | -20.6% | -10.3% | -13.7% | +0.7% | -0.0% | +2.9% |
| L4 - EMA (= Sculley mini-batch) | -10.2% | +5.7% | +0.6% | +0.6% | -14.1% | +20.1% | +101.0% | +126.6% | -10.6% | +4.4% | -1.3% | +1.5% |
| L5 - mini-batch (= Lloyd) | -0.0% | +17.7% | +0.0% | -0.0% | +0.0% | +39.9% | +0.0% | +12.4% | +0.0% | +16.8% | -0.0% | +2.9% |

 Step impact (Δ vs previous rung, % of held-out MSE; positive = removing it hurt):

| removed | gmm-d32 | uniform-d64 | gmm-d2 | covtype | covtype-minmax | mnist |
|---|---|---|---|---|---|---|
| revival | +7.8% | +1.0% | +0.0% | -61.2% | -0.6% | +0.4% |
| closing means | +0.0% | -0.0% | -0.3% | -0.0% | -0.0% | -0.0% |
| cosine schedule | +0.4% | +0.3% | -2.0% | +1.3% | +1.0% | +2.3% |
| EMA (= Sculley mini-batch) | +3.4% | -0.7% | +20.6% | +160.1% | +3.6% | -1.3% |
| mini-batch (= Lloyd) | +11.3% | -0.6% | +16.5% | -49.7% | +11.9% | +1.3% |

### T = 3 epochs

| rung | gmm-d32 vs km@T | vs km@50 | uniform-d64 vs km@T | vs km@50 | gmm-d2 vs km@T | vs km@50 | covtype vs km@T | vs km@50 | covtype-minmax vs km@T | vs km@50 | mnist vs km@T | vs km@50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 e-means | -14.4% | -7.3% | +0.1% | +0.0% | -24.7% | -0.1% | +95.9% | +112.8% | -8.5% | -0.3% | -1.9% | -0.1% |
| L1 - revival | -7.0% | +0.7% | +1.1% | +1.0% | -24.7% | -0.1% | -18.8% | -11.8% | -8.9% | -0.8% | -1.6% | +0.3% |
| L2 - closing means | -7.0% | +0.7% | +1.1% | +1.0% | -24.8% | -0.2% | -18.8% | -11.8% | -8.9% | -0.8% | -1.5% | +0.3% |
| L3 - cosine schedule | -7.0% | +0.7% | +1.3% | +1.3% | -25.8% | -1.5% | -17.8% | -10.7% | -7.8% | +0.5% | +0.7% | +2.6% |
| L4 - EMA (= Sculley mini-batch) | -3.1% | +5.0% | +0.6% | +0.5% | -10.4% | +18.9% | +105.5% | +122.9% | -4.6% | +3.9% | -0.5% | +1.4% |
| L5 - mini-batch (= Lloyd) | -0.2% | +8.1% | +0.0% | -0.1% | -0.0% | +32.8% | +0.6% | +8.9% | +0.0% | +8.9% | +0.0% | +1.9% |

 Step impact (Δ vs previous rung, % of held-out MSE; positive = removing it hurt):

| removed | gmm-d32 | uniform-d64 | gmm-d2 | covtype | covtype-minmax | mnist |
|---|---|---|---|---|---|---|
| revival | +8.6% | +1.0% | +0.0% | -58.6% | -0.5% | +0.3% |
| closing means | +0.0% | -0.0% | -0.1% | +0.0% | -0.0% | +0.0% |
| cosine schedule | -0.0% | +0.3% | -1.3% | +1.3% | +1.3% | +2.3% |
| EMA (= Sculley mini-batch) | +4.3% | -0.7% | +20.7% | +157.2% | +3.4% | -1.2% |
| mini-batch (= Lloyd) | +3.0% | -0.6% | +11.6% | -50.4% | +4.8% | +0.5% |

### T = 5 epochs

| rung | gmm-d32 vs km@T | vs km@50 | uniform-d64 vs km@T | vs km@50 | gmm-d2 vs km@T | vs km@50 | covtype vs km@T | vs km@50 | covtype-minmax vs km@T | vs km@50 | mnist vs km@T | vs km@50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 e-means | -10.9% | -7.9% | +0.1% | +0.0% | -21.6% | -2.1% | +96.4% | +105.8% | -4.5% | -0.4% | -1.1% | -0.1% |
| L1 - revival | -4.3% | -1.2% | +0.6% | +0.6% | -21.6% | -2.1% | -16.7% | -12.7% | -5.6% | -1.5% | -1.0% | +0.0% |
| L2 - closing means | -4.3% | -1.1% | +0.6% | +0.6% | -21.7% | -2.2% | -16.7% | -12.7% | -5.5% | -1.5% | -1.0% | +0.0% |
| L3 - cosine schedule | -3.2% | -0.0% | +1.0% | +1.0% | -21.9% | -2.5% | -13.6% | -9.4% | -1.7% | +2.6% | +1.7% | +2.8% |
| L4 - EMA (= Sculley mini-batch) | +0.9% | +4.2% | +0.2% | +0.2% | -3.7% | +20.3% | +94.8% | +104.2% | +0.4% | +4.7% | +0.2% | +1.2% |
| L5 - mini-batch (= Lloyd) | -0.2% | +3.1% | +0.0% | -0.0% | -0.0% | +24.9% | +1.0% | +5.6% | +0.0% | +4.3% | -0.0% | +1.1% |

 Step impact (Δ vs previous rung, % of held-out MSE; positive = removing it hurt):

| removed | gmm-d32 | uniform-d64 | gmm-d2 | covtype | covtype-minmax | mnist |
|---|---|---|---|---|---|---|
| revival | +7.3% | +0.6% | +0.0% | -57.7% | -1.1% | +0.1% |
| closing means | +0.0% | -0.0% | -0.1% | +0.0% | +0.0% | +0.0% |
| cosine schedule | +1.2% | +0.4% | -0.3% | +3.8% | +4.1% | +2.8% |
| EMA (= Sculley mini-batch) | +4.2% | -0.8% | +23.4% | +126.2% | +2.1% | -1.5% |
| mini-batch (= Lloyd) | -1.0% | -0.2% | +3.8% | -45.4% | -0.4% | -0.2% |

### T = 10 epochs

| rung | gmm-d32 vs km@T | vs km@50 | uniform-d64 vs km@T | vs km@50 | gmm-d2 vs km@T | vs km@50 | covtype vs km@T | vs km@50 | covtype-minmax vs km@T | vs km@50 | mnist vs km@T | vs km@50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 e-means | -10.2% | -9.3% | +0.0% | +0.0% | -16.8% | -3.8% | +108.3% | +112.5% | -2.4% | -0.6% | -0.6% | -0.2% |
| L1 - revival | -4.2% | -3.3% | +0.3% | +0.3% | -16.8% | -3.8% | -14.3% | -12.6% | -2.7% | -1.0% | -0.6% | -0.1% |
| L2 - closing means | -4.2% | -3.2% | +0.3% | +0.3% | -16.8% | -3.7% | -14.3% | -12.6% | -2.7% | -1.0% | -0.6% | -0.1% |
| L3 - cosine schedule | +10.3% | +11.4% | +5.1% | +5.0% | -16.6% | -3.6% | -13.2% | -11.5% | -1.4% | +0.4% | +6.7% | +7.2% |
| L4 - EMA (= Sculley mini-batch) | +2.9% | +4.0% | +0.1% | +0.0% | +2.8% | +18.9% | +83.7% | +87.3% | +2.6% | +4.5% | +0.7% | +1.1% |
| L5 - mini-batch (= Lloyd) | -0.0% | +1.0% | +0.0% | -0.0% | +0.0% | +15.7% | +1.1% | +3.0% | +0.0% | +1.8% | +0.0% | +0.4% |

 Step impact (Δ vs previous rung, % of held-out MSE; positive = removing it hurt):

| removed | gmm-d32 | uniform-d64 | gmm-d2 | covtype | covtype-minmax | mnist |
|---|---|---|---|---|---|---|
| revival | +6.7% | +0.3% | +0.0% | -58.9% | -0.4% | +0.0% |
| closing means | +0.0% | -0.0% | +0.0% | +0.0% | +0.0% | -0.0% |
| cosine schedule | +15.1% | +4.7% | +0.2% | +1.3% | +1.4% | +7.3% |
| EMA (= Sculley mini-batch) | -6.7% | -4.7% | +23.3% | +114.5% | +4.1% | -5.6% |
| mini-batch (= Lloyd) | -2.9% | -0.1% | -2.7% | -42.9% | -2.5% | -0.7% |

## Leave one out: remove a single ingredient from full e-means

Mean paired % change in held-out MSE vs full e-means at the same T (positive = the ingredient helps). Last row: full e-means vs faiss k-means at the same T.


### T = 1 epochs

| removed | gmm-d32 | uniform-d64 | gmm-d2 | covtype | covtype-minmax | mnist |
|---|---|---|---|---|---|---|
| -revival | +8.5% | +1.7% | +0.0% | -1.5% | -0.2% | +0.8% |
| -closing | -2.1% | +0.0% | -3.6% | -4.3% | -1.3% | -0.1% |
| -cosine | +0.2% | +0.2% | -1.9% | +17.1% | +0.9% | -0.1% |
| -ema (Sculley) | +3.9% | +0.1% | +8.9% | +6.0% | +1.9% | +0.6% |
| -batch_law (B=1024) | -1.1% | +0.0% | +0.4% | +9.4% | +1.8% | +0.0% |
| -step_law (alpha0=B/n) | +3.1% | +0.1% | +9.1% | +5.9% | +2.1% | +0.7% |
| -shuffle | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| *e-means vs k-means@T* | -32.1% | -0.2% | -28.6% | -47.6% | -29.6% | -4.7% |

### T = 2 epochs

| removed | gmm-d32 | uniform-d64 | gmm-d2 | covtype | covtype-minmax | mnist |
|---|---|---|---|---|---|---|
| -revival | +7.8% | +1.0% | +0.0% | -61.2% | -0.6% | +0.4% |
| -closing | -0.0% | -0.0% | -0.3% | -5.9% | -0.0% | -0.0% |
| -cosine | -1.9% | +0.0% | -2.0% | +0.5% | -0.1% | +0.0% |
| -ema (Sculley) | +6.2% | +0.0% | +16.5% | -14.7% | +3.1% | +0.9% |
| -batch_law (B=1024) | -0.6% | -0.0% | +1.0% | -0.7% | +1.5% | -0.1% |
| -step_law (alpha0=B/n) | +5.9% | +0.1% | +16.2% | -14.4% | +3.0% | +0.9% |
| -shuffle | -0.1% | +0.0% | -0.1% | -7.0% | +0.1% | -0.0% |
| *e-means vs k-means@T* | -19.7% | +0.0% | -27.2% | +101.1% | -14.1% | -2.7% |

### T = 3 epochs

| removed | gmm-d32 | uniform-d64 | gmm-d2 | covtype | covtype-minmax | mnist |
|---|---|---|---|---|---|---|
| -revival | +8.6% | +1.0% | +0.0% | -58.6% | -0.5% | +0.3% |
| -closing | +0.0% | -0.0% | -0.1% | +0.0% | -0.0% | +0.0% |
| -cosine | -2.1% | -0.0% | -2.4% | +2.6% | +0.2% | +0.0% |
| -ema (Sculley) | +7.7% | +0.0% | +17.5% | -9.0% | +3.4% | +0.9% |
| -batch_law (B=1024) | -0.4% | -0.0% | +0.0% | -0.5% | +1.7% | +0.0% |
| -step_law (alpha0=B/n) | +7.3% | +0.0% | +16.8% | +10.8% | +3.1% | +0.8% |
| -shuffle | +0.2% | +0.0% | -0.1% | -0.7% | -0.2% | +0.0% |
| *e-means vs k-means@T* | -14.4% | +0.1% | -24.7% | +95.9% | -8.5% | -1.9% |

### T = 5 epochs

| removed | gmm-d32 | uniform-d64 | gmm-d2 | covtype | covtype-minmax | mnist |
|---|---|---|---|---|---|---|
| -revival | +7.3% | +0.6% | +0.0% | -57.7% | -1.1% | +0.1% |
| -closing | +0.0% | -0.0% | -0.1% | +2.4% | +0.0% | -0.0% |
| -cosine | -3.7% | -0.0% | -1.5% | +8.5% | -0.4% | +0.0% |
| -ema (Sculley) | +9.1% | -0.0% | +21.0% | -15.1% | +4.4% | +1.0% |
| -batch_law (B=1024) | -1.1% | +0.0% | -0.6% | -16.3% | +1.9% | -0.2% |
| -step_law (alpha0=B/n) | +8.4% | -0.0% | +19.1% | -17.3% | +3.7% | +0.8% |
| -shuffle | +0.5% | +0.0% | +0.4% | +2.8% | -0.1% | -0.0% |
| *e-means vs k-means@T* | -10.9% | +0.1% | -21.6% | +96.4% | -4.5% | -1.1% |

### T = 10 epochs

| removed | gmm-d32 | uniform-d64 | gmm-d2 | covtype | covtype-minmax | mnist |
|---|---|---|---|---|---|---|
| -revival | +6.7% | +0.3% | +0.0% | -58.9% | -0.4% | +0.0% |
| -closing | +0.0% | -0.0% | +0.0% | +0.0% | +0.0% | -0.0% |
| -cosine | -7.2% | +0.1% | -1.2% | -39.2% | +0.1% | +0.2% |
| -ema (Sculley) | +11.2% | -0.0% | +21.7% | +1.5% | +4.4% | +1.0% |
| -batch_law (B=1024) | -1.8% | +0.0% | -0.5% | -0.3% | +1.7% | -0.1% |
| -step_law (alpha0=B/n) | +9.7% | -0.0% | +17.3% | -58.2% | +2.9% | +0.6% |
| -shuffle | +1.1% | +0.0% | -0.2% | +0.1% | +0.8% | -0.0% |
| *e-means vs k-means@T* | -10.2% | +0.0% | -16.8% | +108.3% | -2.4% | -0.6% |
