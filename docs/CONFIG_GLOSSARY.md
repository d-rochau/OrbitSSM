# Config glossary: field, paper symbol, meaning

Field names of `SSDMimoConfig` and `HybridMambaConfig` are explanatory; the symbols are those of the paper
(*Diagonal Equivalence in State Space Models*, Section 5, "Extending dynamics and routing within the
diagonal-equivalent class"). The same table is condensed in each config class docstring
(`help(orbit_ssm.SSDMimoConfig)`).

Per subsystem the recurrence is `h_t = A_t h_{t-1} + b_t`, `y_t = C_t h_t + D_t u_t` with state `h_t` in R^N,
input `u_t` in R^R and output `y_t` in R^O; the drive width `d = H P` is split into `H` heads (index eta) of `P`
coordinates, and the selected step is `Delta_t = softplus(. + dt_bias)`.

## Shared fields

| config field | paper symbol | meaning |
|---|---|---|
| `d_model` | | layer (block) width |
| `d_state` | N | state rows per subsystem |
| `d_inner`, `expand` | d = H P | drive width; `d_inner = int(d_model * expand)` |
| `n_heads` | H (eta) | heads |
| `d_head` | P | coordinates per head (SSD-MIMO: `P = G R`) |
| `d_ssm_in` | R | input rank: inputs per subsystem, `B_t` in R^(N x R); R = 1 is SISO |
| `d_ssm_out` | O | output rank: outputs per subsystem, `C_t` in R^(O x N) |
| `n_groups` | G | channels (groups) per head, SSD-MIMO only; a scan head (eta, r) shares one scalar decay |
| `d_conv` | k_conv | width of the causal convolution; it sets the window `l = k_conv + 1` with the two-endpoint write |
| `write_rule="trapezoid"` | two-endpoint write, weight lambda_t | `h_t = A_t h_{t-1} + Delta_t [lambda_t v_t + (1 - lambda_t) A_t v_{t-1}]` (Mamba-3's rule) |
| `write_rule="euler"` | exponential Euler, lambda = 1 | `b_t = Delta_t v_t` (Mamba-1 and Mamba-2 specializations) |
| `trap_proj_bias` | b_lambda | bias of the projection that selects lambda_t |
| `skip`, `skip_sharing` | D | the D term, `y_t = C_t h_t + D_t u_t`; sharing sets its shape |
| `proj_source` | s_t | coefficient source: `"layer_input"` reads the block input `x_t` (Mamba-2/3), `"main_branch"` the post-convolution drive `u_t` (Mamba-1) |
| `dt_min`, `dt_max`, `dt_init_floor` | Delta_t | range of the initial step and its floor |
| `out_norm` | output RMS normalization | RMSNorm before the output projection (off by default; on in the Mamba-2 preset) |
| `bc_conv` | | short convolution on the B/C projections (Mamba-2's joint xBC convolution) |

## Decay laws (`row_blocks[i]["decay"]`)

| config value | paper name | log-decay per step | families |
|---|---|---|---|
| `head_dynamic` | SSD dynamic; HM flat dynamic | `a_t = -max(act(W_a s_t + b_a), 1e-4)`, per head and token | SSD-MIMO, HybridMamba |
| `head_static` | SSD static; HM flat static | `a = -exp(A_log)`, per head | SSD-MIMO, HybridMamba |
| `coord_static` | HM static | `a = -exp(A_log)`, per (head, channel, state row) | HybridMamba |
| `coord_dynamic` | HM dynamic | `a_t = -act(W_a s_t)`, per state row or head | HybridMamba |
| `coord_potential` | HM potential | `log alpha = Delta a + c (mu_t - mu_{t-1})`, bounded selected potential | HybridMamba |

The applied decay is `alpha = exp(Delta a)` (potential law: see the `log alpha` entry). A block with
`complex=True` rotates; `rows` / `frac` size the block (the sizes sum to `d_state`).

| config field | paper symbol | meaning |
|---|---|---|
| `a_act`, `dyn_act` | act | rate activation; `"heavy_tail"` is `ht(v) = 1 + v` for `v >= 0`, `1 / (1 - v)` otherwise |
| `a_proj_bias` | b_a | bias of the selected-rate projection (flat dynamic law) |
| `a_floor` | the `1e-4` floor | lower clamp on the flat dynamic rate |
| `mu_cap` | mu_cap | bound of the potential law (default 4); the gain is `G_mu sigmoid(.)`, `G_mu = (1/2) log mu_cap` |
| `mu_res`, `dyn_res` | | resolution of the potential / dynamic-rate projections: per state row (`"n"`) or per head (`"h"`) |

## Maps, gates and rotation

| config field | paper symbol | meaning |
|---|---|---|
| `b_norm`, `c_norm` | RMS normalization of the base maps | state-axis RMSNorm of `B-hat_t`, `C-hat_t` (on by default) |
| `bc_bias`, `bc_bias_init` | beta, gamma | static additive biases of the write and read maps (granularity: `"head"`, `"shared"`, `"none"`) |
| `b_gate`, `c_gate` | phi_B, phi_C | write and read gates, `B-bar_t = phi_B(s_t) * B-hat_t + beta` (off unless enabled) |
| `bc_gate_act` | phi | gate activation, by default `phi(v) = 1 + v` |
| `r_mix` | zeta | sibling mix that combines the R readouts (SSD-MIMO, R > 1) |
| `learn_theta_freq` | omega_k = pi tanh(phi_k) | learn the pair frequencies (`theta_freq` stores phi_k; initial omega = 0.999 pi) |
| `use_theta_h` | nu | static read-phase offset per rotating pair, carried on the read side |
| `theta_per_head` | q_(k,t) | one angle feature per head instead of a head-shared one |
| `omega_layout` | omega | shape of the frequency: per pair, per (head, pair) or per head |
| `theta_init` | W_theta | init of the angle-feature projection |
| rows with `complex=True` | K_rot | rotating pairs; pair k is `(2k, 2k+1)` and carries `A_t|_k = alpha_(k,t) R(theta_(k,t))`, `theta_(k,t) = -omega_k tanh(q_(k,t)) Delta_t` |
