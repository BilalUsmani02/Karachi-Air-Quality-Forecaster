---
title: "Karachi Air-Quality Forecaster: A From-Scratch NumPy Neural Network for Day-Ahead PM2.5 Forecasting, Benchmarked Against PyTorch and scikit-learn"
subtitle: "PAI (CT-264) — Project Report"
author: "Group members: [to be added]"
date: "November 2026"
---

# Abstract {-}

Karachi often breathes air with fine-particle (PM2.5) levels well above health guidelines, but residents can only see what the air is like now, not what the coming day will bring. This project forecasts the 24-hour average PM2.5 for the coming day at each of 14 low-cost sensor sites run by The Urban Unit, with the forecast issued at 00:00 Pakistan time (PKT). The inputs are the sensors' own recent history, recorded weather (ERA5 via Open-Meteo) and modelled air quality (CAMS via Open-Meteo).

The core of the project is a multi-layer neural network written from scratch in NumPy: forward pass, backpropagation, mean-squared-error and Huber losses, L2 regularisation, mini-batch SGD and Adam, He and Xavier initialisation, and early stopping. A numerical gradient check confirms the hand-derived gradients (worst relative error $7.8 \times 10^{-10}$). A PyTorch twin, trained from the same initial weights on the same mini-batches, gives predictions that differ from the NumPy network by at most $1.4 \times 10^{-14}$ µg/m³ on 442 test rows.

The network was then compared with persistence, Ridge regression, Random Forest and Gradient Boosting on the same eight rolling-origin cross-validation folds and one final test period that was scored only once. On the final test (1 September to 8 November 2026) the tuned NumPy network reached a mean absolute error (MAE) of 4.98 µg/m³, slightly behind Gradient Boosting (4.73), and ahead of Random Forest (5.37), Ridge (5.60) and persistence (5.82). An equal-weight average of Gradient Boosting and the NumPy network was best overall (MAE 4.63, R² 0.861, 20.5% better than persistence) and is the production model. SHAP explains each forecast in plain English, and a Streamlit dashboard shows forecasts, informational health alerts, rule-based advice and the reasons behind each number.

# Introduction

## The problem

PM2.5 is the mass of airborne particles smaller than 2.5 micrometres per cubic metre of air. These particles reach deep into the lungs and are linked to heart and lung disease, with children, older people and people with asthma most at risk (WHO, 2021). In Karachi, smog builds up in winter and clears during the summer monsoon. The Urban Unit runs a network of low-cost sensors in the city, and their readings are published through OpenAQ. These readings tell people what the air was like an hour ago. A forecast lets people plan: when to exercise, when to keep windows shut, and when a sensitive family member should stay indoors.

## The approved proposal

The proposal approved for this course read:

> "AI system that forecasts Karachi's air pollution level for the next 24 hours using historical air-quality and weather data. A neural network will be built from scratch in NumPy, applying linear algebra and calculus to show how the model learns. It will be validated against PyTorch and compared with standard machine learning models. Explainable AI will highlight the main factors behind each prediction, and a Streamlit dashboard will display forecasts and health alerts."

There is no separate PAI brief. The project overview submitted with the proposal (`docs/briefs/PAI_Overview_original.docx`) describes four parts: the forecaster, the comparison, the explanations and the web app. It lists three things as out of scope: cities other than Karachi, pollutants other than PM2.5, and "official health warnings or alerts". Section 8 explains how the dashboard's health alerts stay inside that limit.

## Objectives

1. Forecast the 24-hour average PM2.5 for the next day at each sensor, issued at 00:00 PKT.
2. Build the network from scratch in NumPy and derive every step of its learning with linear algebra and calculus.
3. Prove the implementation correct twice: with a numerical gradient check, and with an independent PyTorch twin.
4. Compare the network fairly with standard machine-learning models, on the same data splits and the same metrics.
5. Explain each forecast with SHAP.
6. Deliver the forecasts, health alerts and explanations in a Streamlit dashboard.

## What was delivered

Every promise in the proposal was built and tested; Appendix A maps each one to its code, results and section. The NumPy network (Section 3) is verified (Section 4), tuned (Section 5), compared (Section 6) and forms half of the production ensemble, which SHAP (Section 7) and the dashboard (Section 8) build on.

# Data

## Sources

| # | Source | What it gives | Role | Raw rows |
|---|---|---|---|---|
| D1 | The Urban Unit AirGradient sensors, via the OpenAQ API v3 | Hourly PM2.5 at 14 sites, Jun 2025 to Nov 2026 | Target, and its own lag features | 102,110 |
| D2 | Open-Meteo Historical Weather API (ERA5 reanalysis; Hersbach et al., 2020) | Hourly temperature, humidity, dew point, pressure, rain, wind, boundary-layer height | Weather features | 186,816 |
| D3 | Open-Meteo Air Quality API (CAMS) | Hourly modelled PM2.5, PM10, NO₂, dust, aerosol optical depth | Modelled-pollution features | 186,816 |
| D4 | US EPA AQI guidance and WHO 2021 guidelines | Band breakpoints and health advice text | Bands and advice rules | 22 rules |

All three time-series sources are licensed CC BY 4.0 and were downloaded on 8 November 2026. Each download is recorded in a manifest with its endpoint, coordinates and row count. Fourteen sensors have at least 150 complete days. The proposal's overview named 9 sensors, the set visible through the sensor network's live-reading feed at the time; querying the full historical archive later turned up 14 Urban Unit sensors in Karachi with enough history to use, so the project covers all 14. Typically seven to nine report on a given day (eight on average), because several sites have long gaps. Weather and CAMS data start a month before the sensors, so that features that look back in time exist from the first day.

The sensors agree closely: daily means correlate between 0.911 and 0.992 across pairs. CAMS returns one value for the whole city, so it acts as a city-wide signal. Daily PM2.5 falls on warmer (r = −0.652) and windier (r = −0.627) days, mostly because the monsoon is warm, windy and clean.

## Cleaning

`src/data/clean.py` puts every source on one hourly grid in the Asia/Karachi time zone, then applies four rules to each sensor in order. It uses only past and present data, so the same code can run live:

1. Negative values are removed.
2. Spikes are removed: values above 500 µg/m³, or above 100 µg/m³ and more than five times the median of the previous 24 hours.
3. Flat lines are removed: the same value repeated for 24 hours or more is treated as a stuck sensor.
4. Gaps of up to 3 hours are filled by straight-line interpolation. Longer gaps stay missing.

No humidity correction was applied. Low-cost sensors can read high in humid air, but the exploration found the opposite pattern: PM2.5 was lower when relative humidity was above 80% (35.6 against 50.2 µg/m³).

## Target and features

The target for sensor $s$ on day $D$ is the mean of its 24 hourly readings on day $D$. A day counts only if at least 18 of the 24 hours are valid, which follows the EPA's completeness convention. The forecast is issued at 00:00 on day $D$. So "the next 24 hours" in the proposal means exactly the calendar day that starts at the issue time. A 24-hour average also matches how the EPA defines its PM2.5 bands.

`src/data/features.py` builds one row per sensor-day with 58 input features:

- **PM2.5 history (11):** the readings 1, 2, 3, 6, 12 and 24 hours before issue; 72-hour and 168-hour means; the previous day's standard deviation and mean; and the previous day's mean across all sensors in the city.
- **Calendar (5):** day of year and day of week as sine and cosine pairs, and the month.
- **Weather and CAMS (28):** 14 daily values for the day before (`past_*`) and the same 14 for the forecast day (`fc_*`). Wind direction is turned into east-west and north-south components (u and v), because 359° and 1° are neighbours, not opposites.
- **Site (14):** one on/off (one-hot) column per sensor.

After cleaning, the feature table holds 4,027 labelled sensor-days.

**Leakage rule.** Leakage means letting the model see information it would not have at forecast time. Every history feature for day $D$ uses only hours before 00:00 on day $D$. Only the `fc_*` columns use hours inside day $D$. In training and evaluation those columns hold the recorded weather for day $D$, standing in for a weather forecast. This is called *perfect prognosis*. The live app uses Open-Meteo's actual forecast for day $D$ instead, so live accuracy will be somewhat lower than the scores in this report (Section 10).

**Scaling.** `src/models/dataset.py::Preprocessor` fills missing values with the training median and standardises each column to mean 0 and standard deviation 1. Site columns pass through unscaled. The preprocessor is fitted on the training rows of each split only.

## Splits

Rows are split by date, never at random, and all sensors share the same cut dates (`src/data/split.py::project_splits`).

- **Cross-validation (CV):** eight rolling-origin folds (Tashman, 2000). In each fold the model learns from all days up to one day before a 30-day block and is then scored on that block; the skipped day keeps training and test rows apart. The blocks run from 1 January to 28 August 2026, so they cover winter smog, the spring transition and the monsoon. Pooled together, the eight test blocks hold 1,458 sensor-days over 240 dates. Every model setting was chosen on these folds.
- **Final test:** 1 September to 8 November 2026, 442 sensor-days over 69 dates. Each model was trained on all 3,585 labelled rows before 1 September and scored once, after every choice had been frozen. The script `src/evaluate/final_test.py` refuses to run a second time (unless `--force`).

A single summer validation block was considered and rejected: June to August 2026 contained no Unhealthy days at all, so it could not show how a model behaves in smog.

![The end-to-end pipeline. Blue boxes are the PAI focus: the models and their tuning.](docs/report/figures/pipeline.png)

# The neural network from scratch

The network lives in `src/models/numpy_nn/` (about 1,040 lines across eight files). It uses only NumPy arrays and matrix operations (Harris et al., 2020), with no deep-learning library. This section derives every step the code runs.

## Architecture and shapes

A mini-batch is a matrix with one row per sensor-day. The notation is:

| Symbol | Shape | Meaning |
|---|---|---|
| $n$ | | rows in the mini-batch (32, smaller for the last batch of an epoch) |
| $L$ | | number of dense layers ($L-1$ hidden layers plus one output layer) |
| $d_l$ | | width of layer $l$; $d_0$ is the number of inputs, $d_L = 1$ |
| $X = A^{(0)}$ | $n \times d_0$ | the input batch, already imputed and scaled |
| $W^{(l)}$ | $d_{l-1} \times d_l$ | weights of layer $l$ |
| $b^{(l)}$ | $d_l$ | biases of layer $l$ |
| $Z^{(l)}$, $A^{(l)}$ | $n \times d_l$ | pre-activation and activation of layer $l$ |
| $P = Z^{(L)}$ | $n \times 1$ | prediction |
| $y$ | $n \times 1$ | target (standardised) |
| $\mathbf{1}$ | $n \times 1$ | a column of ones |

The tuned network has one hidden layer: 55 inputs, 32 ReLU units and one output, so $L = 2$, $d_0 = 55$, $d_1 = 32$, $d_2 = 1$. It has $55 \times 32 + 32 + 32 + 1 = 1{,}825$ trainable numbers. The code supports any number of hidden layers. The untuned default was two hidden layers of 64 and 32 units.

## Forward pass

For each layer $l = 1, \dots, L$:

$$Z^{(l)} = A^{(l-1)} W^{(l)} + \mathbf{1}\, b^{(l)\top}$$

The shapes multiply as $(n \times d_{l-1})(d_{l-1} \times d_l) = n \times d_l$. The term $\mathbf{1} b^{\top}$ adds the same bias row to every example, which NumPy does by broadcasting. Each hidden layer then passes every element through a non-linear function $\varphi$, while the output layer stays linear:

$$A^{(l)} = \varphi\left(Z^{(l)}\right) \quad (l < L), \qquad P = Z^{(L)}.$$

Without $\varphi$, the layers would multiply out into one matrix, and the network would be no more than a linear model.

## Activations

Three activations are implemented in `layers.py`. Backpropagation needs each one's derivative:

| Name | $\varphi(z)$ | $\varphi'(z)$ |
|---|---|---|
| ReLU | $\max(0, z)$ | 1 if $z > 0$, else 0 |
| LeakyReLU | $z$ if $z > 0$, else $0.01z$ | 1 if $z > 0$, else 0.01 |
| Tanh | $\tanh z$ | $1 - \tanh^2 z$ |

The ReLU derivative at zero is taken as 0, as in PyTorch. LeakyReLU keeps a small slope for negative inputs, so a unit never stops learning completely. Inverted dropout is also implemented: during training each unit is kept with probability $1-p$ and divided by $1-p$, so no rescaling is needed at prediction time. The tuned network uses $p = 0$.

## Losses and their derivatives

Let the residual be $r = P - y$. Both losses take the mean over the batch, as PyTorch does by default.

**Mean squared error (MSE):**

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} r_i^2, \qquad \frac{\partial \mathcal{L}_{\text{MSE}}}{\partial P} = \frac{2}{n}\, r.$$

**Huber loss** (Huber, 1964) with threshold $\delta > 0$:

$$\ell_\delta(r) = \begin{cases} \tfrac12 r^2 & |r| \le \delta \\ \delta\left(|r| - \tfrac12\delta\right) & |r| > \delta \end{cases}, \qquad \frac{\partial \mathcal{L}_{\text{Huber}}}{\partial P} = \frac{1}{n}\,\operatorname{clip}(r, -\delta, \delta).$$

Huber treats small errors like MSE, but a very large error, such as a smoke-event day, pulls on the weights with a force capped at $\delta/n$. The two pieces meet with equal value and equal slope at $|r| = \delta$. Because the target is standardised, $\delta$ is measured in standard deviations, not µg/m³. Note one detail that matches PyTorch: MSE has no $\tfrac12$ in front, so its gradient carries a factor 2, while Huber's quadratic piece does have the $\tfrac12$.

Both gradients are one line of code each (`losses.py`):

```python
# MSELoss.backward
return 2.0 * self._residual / self._residual.size

# HuberLoss.backward
clipped = np.clip(self._residual, -self.delta, self.delta)
return clipped / self._residual.size
```

## Backpropagation

Backpropagation applies the chain rule one layer at a time, from the output back to the input. Write $\Delta^{(l)} = \partial \mathcal{L} / \partial Z^{(l)}$, an $n \times d_l$ matrix that says how much the loss changes when each pre-activation moves. The recursion starts at the output with $\Delta^{(L)} = \partial \mathcal{L}/\partial P$ from the loss. Then, for $l = L, L-1, \dots, 1$:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = A^{(l-1)\top}\, \Delta^{(l)}, \qquad \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \Delta^{(l)\top} \mathbf{1}, \qquad \frac{\partial \mathcal{L}}{\partial A^{(l-1)}} = \Delta^{(l)}\, W^{(l)\top}$$

and, stepping back across the lower layer's activation,

$$\Delta^{(l-1)} = \frac{\partial \mathcal{L}}{\partial A^{(l-1)}} \odot \varphi'\left(Z^{(l-1)}\right),$$

where $\odot$ is element-wise multiplication. A useful sanity check: every gradient must match the shape of the array it belongs to, so $\partial\mathcal{L}/\partial W^{(l)}$ is $d_{l-1} \times d_l$, exactly like $W^{(l)}$. A product written in the wrong order would not line up. Since one bias vector is added to all $n$ rows, its gradient sums $\Delta^{(l)}$ down the rows.

For the tuned network with $n = 32$: $\Delta^{(2)}$ is $32 \times 1$, so $\partial\mathcal{L}/\partial W^{(2)} = A^{(1)\top}\Delta^{(2)}$ is $32 \times 1$; $\partial\mathcal{L}/\partial A^{(1)}$ is $32 \times 32$; the ReLU mask gives $\Delta^{(1)}$, also $32 \times 32$; and $\partial\mathcal{L}/\partial W^{(1)} = X^{\top}\Delta^{(1)}$ is $55 \times 32$.

**Where the batch mean goes.** Many textbooks write $\partial\mathcal{L}/\partial W = \frac{1}{n} A^\top \Delta$. In this code the $\frac{1}{n}$ is already inside $\Delta^{(L)}$, because the loss gradient divides by $n$. Every later step is linear in $\Delta$, so the factor carries through. The dense layer can therefore sum over rows and still return the average gradient per example. Averaging keeps the step size independent of the batch size.

The dense layer's backward pass is the three equations above, almost word for word (`layers.py`, `Dense.backward`):

```python
def backward(self, grad_out: np.ndarray) -> np.ndarray:
    if self._X is None:
        raise RuntimeError("backward() called before forward()")
    self.W.grad = self._X.T @ grad_out      # dL/dW = A_prev^T  Delta
    self.b.grad = grad_out.sum(axis=0)      # dL/db = column sum of Delta
    return grad_out @ self.W.value.T        # dL/dA_prev = Delta W^T
```

The ReLU backward pass is `grad_out * self._mask`, where the mask records which inputs were positive. The container runs the layers in reverse, and one training step chains everything together (`network.py`, `Sequential.backward` and `MLPRegressorNP._train_step`):

```python
def backward(self, grad_out):
    grad = grad_out
    for layer in reversed(self.layers):
        grad = layer.backward(grad)
    return grad

def _train_step(self, Xb, yb):
    pred = self.network_.forward(Xb, training=True)
    loss = self.loss_fn_.forward(pred, yb)
    self.network_.backward(self.loss_fn_.backward())
    self.optimizer_.step()
    return loss
```

## L2 regularisation

To discourage large weights, which tend to fit noise, the objective adds a penalty on the weight matrices but not the biases:

$$J = \mathcal{L} + \frac{\lambda}{2} \sum_{l=1}^{L} \left\| W^{(l)} \right\|_F^2, \qquad g_W = \frac{\partial \mathcal{L}}{\partial W^{(l)}} + \lambda W^{(l)}, \qquad g_b = \frac{\partial \mathcal{L}}{\partial b^{(l)}}.$$

With plain gradient descent this becomes $W \leftarrow (1 - \eta\lambda) W - \eta\,\partial\mathcal{L}/\partial W$: every step shrinks each weight slightly, which is why L2 is also called *weight decay*. Biases are excluded because shifting a bias does not make the function more wiggly. The tuned network uses $\lambda = 0.1$. The penalty is added once per mini-batch step, and the loss curves in this report show the data loss $\mathcal{L}$ only.

## Optimisers: SGD and Adam

Below, $\theta$ stands for any parameter array, $\eta$ for the learning rate and $g_t$ for the gradient at step $t$ (with $\lambda\theta$ added for weights).

**Mini-batch stochastic gradient descent (SGD):** $\theta \leftarrow \theta - \eta\, g_t$.

**Adam** (Kingma and Ba, 2015) keeps two running averages per parameter, both starting at zero, with every operation applied element by element:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2,$$

$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}, \qquad \theta \leftarrow \theta - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon},$$

with $\beta_1 = 0.9$, $\beta_2 = 0.999$ and $\epsilon = 10^{-8}$, PyTorch's defaults. Here $m$ tracks the average direction of the gradient and $v$ its average squared size, so each weight gets its own step size. Because $m$ and $v$ start at zero, they are too small in the first steps. Dividing by $1-\beta^t$ corrects this *bias*. For example, at $t = 1$ the correction gives $\hat m_1 = g_1$ and $\hat v_1 = g_1^2$, so the first step is about $\eta\,\operatorname{sign}(g_1)$: every weight moves by roughly $\eta$, whatever the size of its gradient. The tuned network uses $\eta = 10^{-3}$.

The L2 term enters $g_t$ before the moments are updated. This is classic L2, as in `torch.optim.Adam(weight_decay=λ)`, not the separate decay step of AdamW. The code (`optimizers.py`, `Adam.step`):

```python
def step(self) -> None:
    self.t += 1
    correction1 = 1.0 - self.beta1**self.t
    correction2 = 1.0 - self.beta2**self.t
    for i, p in enumerate(self.params):
        g = regularised_grad(p, self.l2)            # grad + l2 * W for weights
        self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
        self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g**2
        m_hat = self.m[i] / correction1
        v_hat = self.v[i] / correction2
        p.value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

## Initialisation

Biases start at zero. The starting weights are random, with a spread chosen so the signal keeps about the same size from layer to layer (`init.py`):

- **He initialisation** (He et al., 2015) for ReLU and LeakyReLU hidden layers: $W_{ij} \sim \mathcal{N}\left(0, \tfrac{2}{\text{fan}_{\text{in}}}\right)$, where $\text{fan}_{\text{in}} = d_{l-1}$.
- **Xavier (Glorot) initialisation** (Glorot and Bengio, 2010) for tanh layers and the linear output layer: $W_{ij} \sim \mathcal{U}(-a, a)$ with $a = \sqrt{6/(\text{fan}_{\text{in}} + \text{fan}_{\text{out}})}$, which gives variance $2/(\text{fan}_{\text{in}} + \text{fan}_{\text{out}})$.

Why these spreads? Each pre-activation adds up $\text{fan}_{\text{in}}$ weight-times-input products, so its variance grows roughly as $\text{fan}_{\text{in}}\, \operatorname{Var}(w)\, \mathbb{E}[a^2]$, and a weight variance proportional to $1/\text{fan}_{\text{in}}$ cancels that growth. ReLU outputs zero for about half of its inputs, halving $\mathbb{E}[a^2]$; He's factor 2 restores it. Xavier uses both $\text{fan}_{\text{in}}$ and $\text{fan}_{\text{out}}$, a compromise between a steady forward signal and a steady backward gradient. Weights that start too large make activations and gradients explode; weights that start too small make them vanish.

All randomness comes from one seed (42). It is split into three independent streams for initialisation, batch shuffling and dropout, so a run can be repeated exactly.

## Early stopping on time-ordered validation

Inside each training run, the last 15% of the training rows, in date order, are held back for validation (`estimator_base.py`, `BaseMLPRegressor._split`). In the production run on all 3,585 labelled rows before 1 September 2026, that splits off the most recent 538 rows for validation, leaving 3,047 rows for the gradient updates (`results/numpy_vs_torch/summary.json`, which records both the row count and the 0.15 validation fraction for that run). After each epoch (one pass over the training rows) the validation loss is measured. If it has not improved for 20 epochs, training stops and the weights from the best epoch are restored (`estimator_base.py`, `BaseMLPRegressor.fit`).

The held-back rows are the *latest* ones, not a random 15%. Neighbouring days are strongly correlated, because smog spells last for days and the lag features overlap. A random hold-out would put near-copies of training days into validation, make the validation loss look too good, and stop training too late. Holding out the most recent block copies the real task: learn from the past and judge on the future.

## Target scaling and the residual target

PM2.5 runs from about 5 to 250 µg/m³. The network learns the standardised target $\tilde y = (y - \mu)/\sigma$, with $\mu$ and $\sigma$ taken from the training rows, and `predict` maps the output back to µg/m³. This lets one learning rate work regardless of the units.

The tuned network makes one further change, in the wrapper `variant.py` (`MLPVariant`). Instead of the PM2.5 level, it learns the *residual*: the gap between the target and a straight-line persistence forecast,

$$y_{\text{res}} = y - (a + b\, x_{\text{prev}}),$$

where $x_{\text{prev}}$ is yesterday's average PM2.5 and $(a, b)$ are fitted by least squares on the training rows only. At prediction time the line is added back. The wrapper also drops three calendar features (day-of-year sine and cosine, and month), which leaves 55 inputs. Section 5 explains why both changes were made. In the final training run the residual's standard deviation was 16.8 µg/m³, so the losses in the figures are in units of that size.

# Verification

Two separate checks show that the network is correct. The gradient check tests the calculus. The PyTorch twin tests the whole training procedure.

## Gradient check

A derivative is a slope, so it can be measured using the forward pass alone. Nudge one parameter $w$ by $\varepsilon = 10^{-6}$ in each direction and compare the objective $J$ (a *central difference*):

$$\frac{\partial J}{\partial w} \approx \frac{J(w + \varepsilon) - J(w - \varepsilon)}{2\varepsilon}.$$

The approximation error shrinks with $\varepsilon^2$. This is repeated for every number in every parameter array, and each result is set against the backward-pass gradient through the relative error

$$\text{rel}(a, \hat a) = \frac{\lVert a - \hat a \rVert_2}{\lVert a \rVert_2 + \lVert \hat a \rVert_2}.$$

In 64-bit floating point, a correct gradient gives about $10^{-8}$ or less, while a wrong one gives something near 1. The checked objective is the full $J$, including the L2 penalty, so the test covers the exact gradient the optimiser uses (`gradcheck.py`, `check_network`).

The test network has layers $4 \to 6 \to 5 \to 1$ and is run on 12 random rows with $\lambda = 0.01$, for every activation and both losses. With $\delta = 0.5$, eight of the 12 residuals fall in Huber's linear branch and four in its quadratic branch, so both branches are tested. Running `uv run python -m src.models.numpy_nn.gradcheck` prints:

```
worst relative error: 7.84e-10 (a correct gradient gives ~1e-8 or less; a bug gives ~1)
```

The worst relative error for each configuration, from `results/numpy_vs_torch/gradcheck.csv`, is:

| Loss | ReLU | Tanh | LeakyReLU |
|---|---|---|---|
| MSE | $5.7 \times 10^{-10}$ | $2.3 \times 10^{-10}$ | $6.3 \times 10^{-10}$ |
| Huber ($\delta = 0.5$) | $6.3 \times 10^{-10}$ | $4.0 \times 10^{-10}$ | $7.8 \times 10^{-10}$ |

All 36 parameter arrays (3 activations × 2 losses × 6 arrays) pass, from $9.0 \times 10^{-12}$ at best to $7.8 \times 10^{-10}$ at worst, well below the test threshold of $10^{-6}$. The unit tests also check each layer and loss on its own, and confirm that a deliberately wrong gradient (Huber without the clip) is caught.

## Equivalence with PyTorch

The gradient check proves the derivatives. It does not prove the optimiser, the L2 handling or the batching. For that, the same network was rebuilt in PyTorch (Paszke et al., 2019) in `src/models/torch_nn/model.py`. The twin uses `nn.Linear`, PyTorch's own activations and losses, automatic differentiation (*autograd*) for the gradients and `torch.optim.Adam` for the updates. It reuses none of the NumPy maths.

Some things are deliberately shared, so that any difference can only come from the maths. Both models run the same training loop (`estimator_base.py`) with the same seed. That gives identical initial weights (checked: difference exactly 0) and the same order of mini-batches. Both use 64-bit floats on the CPU, the tuned settings, and the same `MLPVariant` wrapper, so they receive the same 55 inputs and the same residual target. PyTorch stores weights as (outputs × inputs), so the NumPy matrix is loaded transposed.

Both networks were trained on all 3,585 labelled rows before 1 September 2026. Their predictions were then compared on the 442 final-test rows. Only the two models' outputs were compared, without reading the true values, so this does not count as a use of the final test.

| Quantity | NumPy vs PyTorch |
|---|---|
| Prediction difference, max / mean (442 rows) | $1.4 \times 10^{-14}$ / $8.4 \times 10^{-16}$ µg/m³ |
| Prediction difference on training rows, max | $2.8 \times 10^{-14}$ µg/m³ |
| Final weight difference, max (`W1`, `b1`, `W2`, `b2`) | $6.9 \times 10^{-17}$, $8.3 \times 10^{-17}$, $1.1 \times 10^{-16}$, $2.8 \times 10^{-17}$ |
| Per-epoch loss difference, max (training / validation) | $1.1 \times 10^{-16}$ / $1.4 \times 10^{-17}$ |
| Per-step loss difference, max over 3,552 mini-batches | $8.9 \times 10^{-16}$ |
| Epochs run / best epoch | 37 / 17 for both |
| Training time (saved run) | NumPy 2.7 s, PyTorch 14.1 s |

A 64-bit float holds about 16 significant digits, so a difference of $10^{-16}$ in a number of size 1 is rounding noise. The two implementations do the same arithmetic in a slightly different order. They stop at the same epoch and keep the same best epoch, because their validation losses never differ enough to change a comparison.

![Left: training and validation loss per epoch for the NumPy network (wide lines) and PyTorch twin (dashed); the curves lie on top of each other. Right: the absolute difference on a log scale (exact zeros drawn at $10^{-18}$).](results/numpy_vs_torch/loss_curves.png)

Timings vary with machine load (another run measured 0.9 s against 31 s), but NumPy was faster each time. With a 32 × 55 batch, each of the 3,552 steps is dominated by PyTorch's fixed per-call cost (building the autograd graph, optimiser bookkeeping), not by arithmetic. For large networks or on a GPU this would reverse.

# Training and tuning

## The learning curve

![Training and validation loss per epoch of the tuned NumPy network. Training stops at epoch 37; the weights from epoch 17 are kept.](results/numpy_vs_torch/learning_curve.png)

The training loss falls from 0.87 to about 0.37 within six epochs and then levels off. With one hidden layer of 32 units and strong L2 ($\lambda = 0.1$), the network reaches its capacity quickly. The validation loss reaches its lowest value (0.035) at epoch 17 and does not improve over the next 20 epochs, so training stops at epoch 37 and the epoch-17 weights are restored.

The validation loss sits far below the training loss. This does not mean the model does better on new data in general. The validation block is the latest 15% of rows, 3 June to 31 August 2026, which is the clean monsoon season: PM2.5 varied with a standard deviation of 8.7 µg/m³ there, against 33.4 µg/m³ in the training part, which includes winter smog. A single validation block can be very unrepresentative, which is why the model was chosen on eight folds spread across the year rather than on this block.

## Why the untuned network did badly

With default settings (two hidden layers of 64 and 32 units, $\lambda = 10^{-4}$, all 58 features, PM2.5 level as target), the network was the worst model in the first round. Its pooled CV MAE was 11.21 µg/m³, worse even than persistence (8.70). A per-fold diagnosis (`results/tuning/mlp_numpy_diagnosis.csv`) found two causes.

**Unseen seasons.** The data starts in June 2025. When a fold tests March to May 2026, the training rows contain no March-to-May days at all, so those day-of-year values are new to the model. The network read them as winter and forecast too high. In fold 2 its average error (bias) was +21.9 µg/m³: the training target averaged 54.5 µg/m³ but the test block only 34.3. Removing the three calendar features cut that bias to −2.5 µg/m³, and the fold's MAE fell from 22.3 to 8.9. The tree models are less exposed to this, because a tree cannot extend a trend beyond the values it has seen.

**Overfitting.** On folds 0 to 4, the training MAE was about 3 to 6 µg/m³, while the MAE on the held-back validation rows was about 15 to 18. The network was memorising its training days. A smaller network (one layer of 32 units) with a much larger penalty ($\lambda = 0.1$) closed most of that gap.

The third change, the residual target, makes the task easier. Yesterday's average is by far the strongest single predictor (Section 7). With the residual target the network no longer has to relearn that relationship. It only learns how weather and recent changes move today away from a fitted persistence line. The line is fitted on training rows only, so it adds no leakage.

Feature selection and target transformation are normal, documented modelling choices. The comparison stays fair because every model is scored on the same folds, rows and metrics, and on the same once-only final test.

## The search and chosen settings

The search was staged rather than a full grid: first one setting at a time around a baseline, then combinations of the best values. It ran 34 configurations on all eight folds, which took 440.6 seconds in total (`results/tuning/mlp_numpy_trials.csv`). It covered the feature set (all or no calendar), target (level or residual), hidden sizes (16, 32, or 64 and 32), activation (ReLU or tanh), loss (MSE or Huber with $\delta$ of 0.3 or 1), learning rate (0.0003, 0.001, 0.003), batch size (32, 64, 128), L2 (from $10^{-4}$ to 0.1), dropout (0, 0.1, 0.3), early stopping on or off, and the validation fraction (0.15 or 0.25).

The best configuration, saved in `results/tuning/mlp_numpy_best.json` and fixed in `src/models/final.py`, is:

| Setting | Value |
|---|---|
| Inputs | 55 (no calendar features) |
| Target | residual from fitted persistence, standardised |
| Hidden layers | one, 32 ReLU units |
| Loss | MSE |
| Optimiser | Adam, learning rate 0.001, batch size 32 |
| L2 | 0.1 |
| Dropout | 0 |
| Early stopping | last 15% of rows, patience 20, at most 200 epochs |

Its pooled CV scores are MAE 6.63 µg/m³, RMSE 10.58, R² 0.870, and a skill of +23.8% over persistence. Some neighbouring settings scored almost the same: adding dropout of 0.3 gave 6.68. The search is therefore not on a knife edge.

Per fold, the network beat persistence on six of the eight folds. It lost narrowly on fold 5 (31 May to 29 June; 3.29 against 2.80), a clean early-summer block, and on fold 7 (2.34 against 2.25), a clean monsoon block. In both, yesterday's value is already an excellent guess.

## Seed stability

A network's result depends partly on its random starting weights. Re-running the chosen configuration with seeds 0, 1 and 2 gave CV MAEs of 6.57, 6.63 and 6.70 µg/m³. The spread of 0.12 µg/m³ is small next to the gaps between model families, so the ranking in the next section does not depend on a lucky seed.

# Comparison with standard machine-learning models

## The models

Every model is trained and scored by the same code, on the same preprocessed matrix:

- **Persistence:** the forecast for day $D$ is yesterday's 24-hour average. Any useful model must beat this. *Skill* is defined against it as $1 - \text{MAE}_{\text{model}}/\text{MAE}_{\text{persistence}}$.
- **Ridge regression:** linear regression with an L2 penalty, the official baseline (tuned $\alpha = 150$).
- **Random Forest:** an average of 500 decision trees, each trained on a random resample (maximum depth 15, at least 5 rows per leaf, half the features tried at each split).
- **Gradient Boosting:** scikit-learn's histogram-based gradient boosting (Pedregosa et al., 2011), which adds trees one at a time, each correcting the errors of the previous ones (100 rounds, 31 leaves, L2 0.1).
- **NumPy network:** Sections 3 to 5.
- **Ensemble:** the equal-weight average of Gradient Boosting and the NumPy network.

The scikit-learn models were tuned with the same staged search on the same folds. The metrics are MAE, RMSE (root mean squared error, which penalises large misses more), R² (the share of variance explained) and skill.

## Cross-validation results

Pooled over the 1,458 sensor-days of the eight folds:

| Model | MAE (µg/m³) | RMSE | R² | Skill |
|---|---|---|---|---|
| Persistence | 8.70 | 14.70 | 0.749 | — |
| Ridge (tuned, official baseline) | 7.22 | 10.54 | 0.871 | +17.0% |
| Random Forest (tuned) | 6.54 | 10.09 | 0.882 | +24.8% |
| Gradient Boosting (tuned) | 6.30 | 9.81 | 0.888 | +27.5% |
| NumPy network (untuned) | 11.21 | 15.42 | 0.723 | −28.9% |
| NumPy network (tuned) | 6.63 | 10.58 | 0.870 | +23.8% |
| **Ensemble (GB + NumPy network)** | **6.16** | **9.58** | **0.893** | **+29.2%** |

The ensemble's row is the average of the Gradient Boosting and NumPy network CV predictions on the same rows, scored with the same metric code (`results/tuning/ensemble_cv_metrics.json`).

Tuning took the network from the worst model to third among the single models, just behind Random Forest and Gradient Boosting. Its MAE is 0.32 µg/m³ behind Gradient Boosting, 0.09 behind Random Forest, and well ahead of Ridge and persistence.

## The ensemble

The NumPy network and Gradient Boosting make different mistakes: the correlation between their predictions is 0.97, not 1. Averaging them cancels part of each one's errors. Several combinations were tried:

| Equal-weight average | CV MAE | Skill |
|---|---|---|
| All four (Ridge, RF, GB, network) | 6.13 | +29.5% |
| **Gradient Boosting + NumPy network** | **6.16** | **+29.2%** |
| Gradient Boosting alone | 6.30 | +27.5% |
| NumPy network alone | 6.63 | +23.8% |

The four-model row and the 0.97 correlation above are not among the saved tuning outputs used elsewhere in this section; both are recomputed from the same saved per-model predictions by `src/evaluate/ensemble_diagnostics.py` and written to `results/tuning/ensemble_diagnostics.json`. The two-model ensemble was chosen. It is within 0.03 µg/m³ of the four-model average, simpler, has the best band scores of any combination (below), and keeps the from-scratch network in the final system. The weights are equal on purpose: fitting weights on the same folds used to judge them would overfit those folds.

## Final test

Scored once, on 442 sensor-days from 1 September to 8 November 2026 (`results/final_test/metrics.csv`):

| Model | MAE (µg/m³) | RMSE | R² | Skill |
|---|---|---|---|---|
| Persistence | 5.82 | 8.64 | 0.762 | — |
| Ridge (official baseline) | 5.60 | 7.72 | 0.810 | +3.8% |
| Random Forest | 5.37 | 7.80 | 0.806 | +7.8% |
| Gradient Boosting | 4.73 | 6.85 | 0.851 | +18.7% |
| NumPy network | 4.98 | 6.97 | 0.846 | +14.5% |
| **Ensemble (GB + NumPy network)** | **4.63** | **6.61** | **0.861** | **+20.5%** |

The ranking from CV holds, except that the NumPy network moves ahead of Random Forest. To be direct about it: the network on its own is slightly behind Gradient Boosting (4.98 against 4.73 µg/m³). Its value lies elsewhere. It is competitive with a strong, mature library model; it beats Random Forest, Ridge and persistence; and adding it to Gradient Boosting improves every metric (MAE 4.73 to 4.63, R² 0.851 to 0.861). The ensemble's MAE is 17% lower than the Ridge baseline's. All errors are smaller than in CV because September to November was cleaner than January to August.

## Air-quality bands, briefly

The app also sorts each forecast into a US EPA band (US EPA, 2024): Good (up to 9.0 µg/m³), Moderate (to 35.4), Unhealthy for Sensitive Groups (USG, to 55.4), Unhealthy (to 125.4) and Very Unhealthy (above; no Hazardous day occurred, so Hazardous is merged into Very Unhealthy). This is the main task of the companion AIES report, so only the headline is given here (`results/bands/metrics.csv`).

On pooled CV, banding the ensemble's forecast puts 82.6% of sensor-days in the right band, with a macro-F1 (F1 averaged over the five bands) of 0.674. It misses 8.0% of truly Unhealthy-or-worse days, against 18.0% for persistence. The 95% bootstrap interval for that difference, resampling whole dates, is −17.4 to −3.5 percentage points, so the gain is not noise. The NumPy network alone gets 80.7% accuracy and macro-F1 0.659. On the final test the ensemble reached 84.6% accuracy and missed 13 of 88 Unhealthy days (14.8%), against 20 for persistence. That final-test gap lies within noise because the period is only 69 days long. The weak spot is the rare extremes: averaging pulls extreme forecasts towards the middle, so the ensemble catches only 16% of the 25 Very Unhealthy days in CV.

Error is uneven across sites (`results/fairness/ensemble/`): CV MAE ranges from 4.6 µg/m³ at the Climate Action Center to 15.6 at DG House DHA Phase 5, with the dirtiest and most sparsely covered sites doing worst.

# Explainable AI

## Method

SHAP (SHapley Additive exPlanations; Lundberg and Lee, 2017) splits one prediction into a *base value* (the average forecast, if nothing were known about the day) plus one contribution per feature. The contributions add up exactly to the prediction. A positive contribution pushed the forecast up and a negative one pulled it down.

The original plan was to explain the PyTorch twin with SHAP's `DeepExplainer`. That plan predates the choice of production model. The model the app shows is an average of a tree model and the NumPy network, and `DeepExplainer` can only open up one PyTorch network, so it cannot explain that average. Instead, `src/explain/shap_explain.py` wraps the ensemble's own `predict` function and uses SHAP's model-agnostic **permutation explainer**. This explainer estimates each feature's share by switching features in and out along random orderings, against a background of 100 training rows with a fixed seed. Exact SHAP over 58 features would need $2^{58}$ feature subsets. One explanation took 0.80 s on the development laptop, fast enough to run live when a user picks a site.

The additivity property was checked on every example in the report: base value plus contributions matched the model's output to about $10^{-13}$ µg/m³, which is floating-point noise.

For a second view, `TreeExplainer`, which computes SHAP exactly for tree models, was run on the Gradient Boosting member over all 4,027 rows.

## Global importance

![Mean absolute SHAP value per feature for the production ensemble (300-row sample).](results/explain/shap_summary.png)

Ranked by mean absolute SHAP value on a fixed 300-row sample (`results/explain/global_importance.csv`):

| Rank | Feature | Mean \|SHAP\| (µg/m³) |
|---|---|---|
| 1 | Yesterday's average PM2.5 | 11.4 |
| 2 | PM2.5 one hour before issue | 4.9 |
| 3 | Forecast day's CAMS PM2.5 | 3.0 |
| 4 | Forecast day's CAMS NO₂ | 2.8 |
| 5 | Yesterday's PM2.5 variability (standard deviation) | 1.9 |
| 6 | Last three days' average PM2.5 | 1.6 |
| 7 | Forecast day's wind (east-west component) | 1.3 |
| 8 | Forecast day's wind speed | 1.2 |
| 9 | Forecast day's boundary-layer (mixing) height | 1.1 |

Two patterns stand out. Persistence dominates: recent PM2.5 outweighs every weather feature, which matches the strength of the persistence baseline. Weather and the CAMS model then adjust the forecast at the margin. Calm air and a low mixing height trap pollution near the ground, while wind disperses it. This is the information a resident cannot get just by checking the current reading.

The two explainers agree on the top factors but not on their order. `TreeExplainer` on Gradient Boosting alone ranks the last hourly reading first and gives more weight to the seasonal cycle. The ensemble explainer also sees the NumPy network, which has no calendar inputs and learns a residual from persistence, so weight shifts towards yesterday's average. Members that lean on different features make different errors, which is why averaging helps.

## A forecast explained

Sensor S11 (Orangi Pilot Project) on 20 December 2025, a winter smog day:

| | µg/m³ |
|---|---|
| Base value (average forecast) | 49.2 |
| Yesterday's average PM2.5 was high (137.3) | +37.5 |
| The reading an hour before issue was high (195.0) | +19.1 |
| The forecast day's CAMS PM2.5 was high (59.3) | +13.4 |
| The forecast day's CAMS NO₂ was high (55.0) | +11.7 |
| Remaining 54 features combined (e.g. last three days' average +6.9, yesterday's variability −6.6) | +26.4 |
| **Forecast** | **157.3** |

The app turns the top three into sentences, for example: "Yesterday's average PM2.5 was high (137.3 µg/m³), which raised the forecast by about 37 µg/m³." The reverse case also works. For S2 (Zafar Memon DHA) on 10 September 2025, the forecast was 5.0 µg/m³: a clean previous day lowered it by about 17 µg/m³, and 86 mm of rain the day before lowered it by about 7.

SHAP explains the model, not the atmosphere. Correlated features such as the three recent PM2.5 measures share the credit between them, and a large SHAP value for wind means the model's output moved with wind, not that wind was proven to be the cause.

# The Streamlit dashboard

The dashboard (`app/`, run with `uv run streamlit run app/main.py`) has three pages. The Streamlit code only handles layout; all the logic is in plain Python modules in `src/ui/`, so it can be unit-tested.

## Forecast page

This is the home page. From top to bottom it shows:

- a status banner with the forecast date, issue time (00:00 PKT), data source and last update;
- a site picker, with both a map and a drop-down list;
- the forecast value and its colour-coded band;
- a health alert when the band is USG or worse;
- two optional profile switches ("sensitive group?" and "outdoor plans?");
- the advice;
- "Why this forecast?", with the top three SHAP reasons;
- a chart of the last 72 hours of readings next to the forecast.

![The Forecast page: city-wide alert, sites without a forecast, and the site map coloured by forecast band.](docs/report/figures/app_forecast.png)

**Health alerts.** The proposal promised health alerts. The overview lists "official health warnings or alerts" as out of scope. Both are respected: the app shows informational alerts that are clearly marked as not official. `src/ui/health_alert.py` produces two kinds. A site alert appears when the selected site's band is USG or worse, shown as a warning for USG and as an error for Unhealthy or worse. A city-wide line counts how many of today's forecast sites are USG or worse. Every site alert ends with "Informational only — not an official alert." The alerts come from EPA's published band definitions applied to the model's forecast. No government agency issues them, and the app does not claim otherwise.

**Advice.** The advice comes from a small rule-based expert system (`src/expert/`, rules R01 to R22). Rules are applied repeatedly until no new facts appear (*forward chaining*). They turn the band, the change from the previous day and the two profile switches into advice quoted from EPA guidance. Each piece of advice cites its source and shows which rules fired and why. If a forecast sits within 5 µg/m³ below the next band, the advice follows the worse band and says so, while the displayed band stays unchanged. A disclaimer on every advice panel says the guidance is general and not medical advice. The profile switches are kept only for the session and never stored. The expert system is described in full in the AIES report.

![Advice for a sensitive user with outdoor plans: the forecast is near the Unhealthy boundary, so the safety margin gives the more cautious advice and says so; the rule trace sits in the collapsed panel.](docs/report/figures/app_advice.png)

## Model comparison and About pages

The Model comparison page reads the saved final-test files and never retrains. It shows the error table, band accuracy, the NumPy-versus-PyTorch summary, the global SHAP chart and per-site error. The About page describes the data sources, the model version and its training window, the limitations, privacy and the disclaimer.

![The Model comparison page.](docs/report/figures/app_compare.png)

## Working without a network

Live forecasts need three online services. The app copes when they fail. A live refresh runs the same cleaning and feature code as training (`src/live/pipeline.py`). If the fetch fails, the app falls back to the last cached snapshot and shows when it was taken, with a warning if the cache is older than today. If there is no cache, it uses a stored demo day. A sidebar "Demo mode" switch shows any past day offline, which is useful for a classroom demonstration. An "Offline map" switch draws the sensor map without background tiles, and it is switched on by default when the tile server cannot be reached.

# Software engineering

## Structure

The code is split by feature: `src/data/` (download, cleaning, features, splits), `src/models/` (baselines, `numpy_nn/`, `torch_nn/`, the ensemble and the frozen settings in `final.py`), `src/evaluate/` (metrics, CV, tuning, final test, band and fairness reports), `src/classify/`, `src/expert/`, `src/explain/`, `src/live/`, `src/ui/` and `app/`. Files are kept short; the largest NumPy network file is 262 lines.

Every model follows the same small contract: `fit(X, y)` and `predict(X)` on a 64-bit NumPy matrix. So the NumPy network, the PyTorch twin, the scikit-learn models and the ensemble all go through the same CV and test code without special cases. The NumPy network also supports scikit-learn's `get_params` and `set_params`, so the tuner can copy it with new settings.

## Testing

The test suite (`tests/`, pytest) has **798 tests** (`uv run pytest --collect-only -q`) and covers **98%** of the lines in `src/`. A test run fails if coverage drops below 80% (`fail_under = 80` in `pyproject.toml`). The network tests are the most relevant here:

- `test_numpy_nn_layers.py` checks each layer's and each loss's gradient numerically, and checks that a broken gradient is detected.
- `test_numpy_nn_regressor.py` checks training, early stopping, weight restoring, input validation and that the network never changes the caller's arrays.
- `test_torch_nn.py` checks that the NumPy network and the PyTorch twin train identically for each optimiser, loss and activation.
- `test_numpy_gradcheck_cli.py` checks the command-line gradient check.

For example, `uv run pytest -q --no-cov tests/test_numpy_nn_layers.py tests/test_numpy_gradcheck_cli.py` reports `33 passed in 0.42s`. Other tests guard against leakage in the splits and features, check the metrics against hand-computed values, and run the app's pages headless.

## Reproducibility

- **Environment:** `uv` with a lockfile (`uv.lock`) pins every package version. The saved comparison ran on Python 3.12.13, NumPy 2.5.4 and PyTorch 2.14.1 (CPU).
- **Randomness:** one seed (42) in `src/config.py`. The network splits it into independent streams, so the same command gives the same weights.
- **Commands:** each result file is written by a single command (see the appendix). The final test refuses to run twice (unless `--force`), so its score cannot be quietly improved.
- **Data:** raw downloads are never edited; manifests record their source and date.

# Limitations and future work

**Perfect-prognosis weather.** Evaluation used recorded weather for the forecast day; the live app uses a forecast. Live accuracy will be lower by an amount that has not yet been measured. Future work: archive the live forecasts each day and score them once a few months have built up.

**Short history.** About 16½ months of data (20 June 2025 to 8 November 2026) cover each season only once or twice. This caused the unseen-season problem in Section 5 and limits how much any model can learn about the seasonal cycle. Dropping the calendar features was a fix for this data set, not a general rule. With two or more full years, the calendar features should be tested again.

**The network trails Gradient Boosting.** On tabular data of this size, boosted trees are hard to beat, and the NumPy network is about 5% behind on final-test MAE. Its contribution is through the ensemble. Possible improvements include learning-rate schedules, batch normalisation, or one network per season, but each would need the same careful fold-based evaluation.

**Extremes.** Averaging pulls extreme forecasts towards the middle, so Very Unhealthy days are under-forecast. The final test contained no Very Unhealthy days and one Good day, so evidence for those bands comes only from CV.

**Sensors.** The sensors are low-cost devices with no co-located reference monitor in this period to calibrate them against.

**Explanations.** SHAP describes the model, which learned correlations from limited data. It is not a causal analysis, and the permutation explainer is an approximation.

**Scope.** The app's alerts and advice are informational, built from EPA and WHO guidance. They are not official warnings or medical advice, in line with the project overview.

# Conclusion

The project delivered what the proposal promised. A neural network was written from scratch in NumPy, with every part of its learning derived by linear algebra and calculus. Numerical gradients agree with it to within $7.8 \times 10^{-10}$, and a PyTorch twin reproduces its predictions to within $1.4 \times 10^{-14}$ µg/m³.

The untuned network performed worse than persistence. A per-fold diagnosis traced that to unseen seasons and overfitting, and three documented changes fixed it: dropping the calendar features, using a smaller and more regularised network, and learning a residual from persistence. On the once-only final test the tuned network reached an MAE of 4.98 µg/m³. That beats Ridge, Random Forest and persistence and comes close to Gradient Boosting (4.73). Averaged with Gradient Boosting, it gives the best model of all (4.63 µg/m³, R² 0.861), which now runs in the app. SHAP shows that yesterday's pollution drives most of each forecast, with weather and modelled pollution adjusting it. The Streamlit dashboard brings forecasts, informational health alerts, advice and reasons together in one place, and it keeps working offline.

# References {-}

Copernicus Atmosphere Monitoring Service (CAMS). Atmospheric composition forecasts, accessed through the Open-Meteo Air Quality API. https://atmosphere.copernicus.eu

Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. In *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 9, pp. 249–256.

Harris, C. R. et al. (2020). Array programming with NumPy. *Nature*, 585, pp. 357–362.

He, K., Zhang, X., Ren, S. and Sun, J. (2015). Delving deep into rectifiers: surpassing human-level performance on ImageNet classification. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 1026–1034.

Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), pp. 1999–2049.

Huber, P. J. (1964). Robust estimation of a location parameter. *The Annals of Mathematical Statistics*, 35(1), pp. 73–101.

Kingma, D. P. and Ba, J. (2015). Adam: a method for stochastic optimization. In *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*.

Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*, pp. 4765–4774.

OpenAQ. Open air quality data platform, API v3. https://openaq.org (data from The Urban Unit's AirGradient sensors, CC BY 4.0; accessed 8 November 2026).

Open-Meteo. Historical Weather API and Air Quality API. https://open-meteo.com (CC BY 4.0; accessed 8 November 2026).

Paszke, A. et al. (2019). PyTorch: an imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, pp. 8024–8035.

Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, pp. 2825–2830.

Streamlit. Streamlit open-source app framework. https://streamlit.io

Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *International Journal of Forecasting*, 16(4), pp. 437–450.

US Environmental Protection Agency (2024). *Technical Assistance Document for the Reporting of Daily Air Quality — the Air Quality Index (AQI)*. Research Triangle Park, NC: US EPA (reflects the 2024 PM2.5 standard revision). https://document.airnow.gov/technical-assistance-document-for-the-reporting-of-daily-air-quailty.pdf

US Environmental Protection Agency / AirNow (2023). *Air Quality Guide for Particle Pollution*. EPA-452/F-23-002.

World Health Organization (2021). *WHO global air quality guidelines: particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide*. Geneva: WHO.

# Appendix A: Proposal promises and evidence {-}

| Promise | Code | Results | Section |
|---|---|---|---|
| Forecast Karachi's air pollution for the next 24 hours | `src/data/features.py` (target), `src/models/final.py`, `src/live/pipeline.py` | `results/final_test/` | 2, 6 |
| Using historical air-quality and weather data | `src/data/download_*.py`, `clean.py`, `features.py` | `data/raw/*/manifest.json` | 2 |
| Neural network built from scratch in NumPy | `src/models/numpy_nn/` (layers, losses, optimizers, init, network, estimator_base, variant) | `results/tuning/mlp_numpy_best.json` | 3, 5 |
| Applying linear algebra and calculus to show how it learns | `layers.py`, `losses.py`, `optimizers.py`, `gradcheck.py` | `results/numpy_vs_torch/gradcheck.csv`, `learning_curve.png` | 3, 4.1, 5.1 |
| Validated against PyTorch | `src/models/torch_nn/` | `results/numpy_vs_torch/summary.json`, `loss_curves.png` | 4.2 |
| Compared with standard machine-learning models | `src/models/baselines/`, `src/evaluate/` | `results/baselines_cv_pooled.csv`, `results/tuning/`, `results/final_test/` | 6 |
| Explainable AI on the main factors behind each prediction | `src/explain/` | `results/explain/` | 7 |
| Streamlit dashboard with forecasts | `app/`, `src/ui/` | screenshots in Section 8 | 8 |
| Health alerts (informational, not official) | `src/ui/health_alert.py`, `src/expert/` | `tests/test_ui_health_alert.py`, `tests/test_expert_*.py` | 8 |

# Appendix B: How to reproduce {-}

```
uv sync                                                 # install the locked environment
uv run python -m src.data.download_sensors              # D1 (needs an OpenAQ key in .env)
uv run python -m src.data.download_weather --start 2025-05-01      # D2
uv run python -m src.data.download_air_quality --start 2025-05-01  # D3
uv run python -m src.data.build_features                # clean + daily feature table
uv run python -m src.evaluate.run_cv                    # baselines on the 8 CV folds
uv run python -m src.evaluate.tune_classical            # tune Ridge, RF, Gradient Boosting
uv run python -m src.evaluate.tune_mlp                  # tune the NumPy network
uv run python -m src.models.numpy_nn.gradcheck          # gradient check
uv run python -m src.models.torch_nn                    # NumPy vs PyTorch, loss figures
uv run python -m src.evaluate.final_test                # final test (runs once)
uv run python -m src.evaluate.final_test --save-production  # refit on all rows for the app
uv run python -m src.evaluate.band_report               # band evidence
uv run python -m src.explain --features data/processed/features.parquet --bundle models/final_ensemble.pkl
uv run pytest                                           # tests + coverage
uv run streamlit run app/main.py                        # the dashboard
```
