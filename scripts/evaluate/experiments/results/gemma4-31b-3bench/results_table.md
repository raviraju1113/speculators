# Speedup vs `baseline` (decode tok/s ratio)

| benchmark | config | n | decode tok/s | e2e tok/s | ttft (s) | accept_len | accept_rate | speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aime | baseline | 30 | 55.1 | 54.8 | 0.148 | — | — | 1.00× |
|  | assistant_k3 | 30 | 135.8 | 133.6 | 0.174 | 3.595 | 0.8649 | **2.46×** |
|  | assistant_k5 | 30 | 163.6 | 159.9 | 0.192 | 4.785 | 0.7570 | **2.97×** |
| gpqa | baseline | 50 | 55.1 | 54.8 | 0.104 | — | — | 1.00× |
|  | assistant_k3 | 50 | 130.2 | 127.8 | 0.123 | 3.377 | 0.7924 | **2.36×** |
|  | assistant_k5 | 50 | 155.3 | 151.6 | 0.129 | 4.443 | 0.6887 | **2.82×** |
| livecodebench | baseline | 50 | 54.9 | 54.4 | 0.144 | — | — | 1.00× |
|  | assistant_k3 | 50 | 128.0 | 124.5 | 0.162 | 3.450 | 0.8167 | **2.33×** |
|  | assistant_k5 | 50 | 150.0 | 145.3 | 0.167 | 4.505 | 0.7010 | **2.73×** |
