# 3. Methodology

In this section, we outline our detailed mathematical formulations, the baseline architectures adopted, our proposed Time-VLM+ (Model C+) refinement, and the rigorous experimental configurations designed to evaluate full-data, few-shot, and ablation performances.

## 3.1. Original Baselines and Foundation Framework
To establish a rigorous, zero-regression benchmark, we source our baseline architectures directly from their original implementations in the ICML25 TimeVLM repository. Our experimental framework evaluates representations spanning several generations of time-series forecasting paradigms to isolate where foundation models truly hold advantages.

*   **Simple Linear Baselines (DLinear):** DLinear acts as a robustness check against over-parameterized models. It decomposes raw time-series data into trend and remainder (seasonal) components using a moving average kernel, and applies a single linear layer to each component: $\hat{Y} = W_{trend} X_{trend} + W_{season} X_{season}$.
*   **Time-Series Transformers (Autoformer & PatchTST):** Autoformer represents first-generation time-series transformers, replacing standard self-attention with a series decomposition and Auto-Correlation mechanism at the series level. PatchTST represents the state-of-the-art for traditional transformers, dividing the time-series into local sub-sequence patches (Patching) combined with Channel Independence to mitigate multivariate noise.
*   **Multimodal Foundation Models (TimeLLM / Model C):** We adopt TimeLLM (utilizing a frozen LLM core such as GPT-2 or LLaMA) to represent foundation forecasting models. TimeLLM tokenizes time-series patches and uses a textual cross-attention reprogramming layer ($O = \text{Softmax}(Q K^T / \sqrt{d})V$) to align numeric patches with pre-trained semantic word embeddings.

To ensure strict fairness, all baseline architectures retain their exact original layer parameters, patching mechanics (`patch_len=16`, `stride=8`), and identical dimensional tensors (`x_mark`, `x_dec`, `x_mark_dec`). We generate the necessary timestamp positional embeddings dynamically during the forward pass to satisfy their rigid architectural signatures without modifying the source files.

## 3.2. Proposed Refinement: Time-VLM+ (Model C+)
Instead of deploying a highly-parameterized architecture from scratch, we propose **Time-VLM+**, a lightweight time-series-aware refinement that mathematically wraps the foundation Model C. Time-VLM+ incorporates a two-pronged mechanism to provide explicit temporal grounding dynamically, allowing the frozen LLM to immediately understand global structural contexts that local patching obscures.

### 3.2.1 Authentic Textual Prompt Injection
Foundation VLMs like TimeLLM map sequences into token spaces and append a static generic dataset prompt. In Time-VLM+, we dynamically override this static description. For every batch, the model computes an explicit 12-dimensional semantic descriptor vector ($P \in \mathbb{R}^{12}$) based on the historical context window. 

These descriptors are dynamically interpolated into a dense, natural-language string template (e.g., *"Dataset context: Complex time-series. Mean: 0.04, Median: 0.01, Std: 1.05. Trend slope: 0.0021, Rolling variance: 0.98, Mean diff: 1.12"*). The VLM tokenizes this string using its native vocabulary and feeds it into the cross-attention reprogramming layer. This natively shifts the keys ($K$) and values ($V$) in the attention equations to reflect grounded temporal truths, rather than abstract word embeddings.

### 3.2.2. Autoregressive Statistical Bypass
Large, frozen language models require significant epochs to alter their cross-attention bounds based solely on text, resulting in sluggish convergence compared to explicit linear mapping. To mitigate this, Time-VLM+ implements an **Autoregressive Statistical Bypass**. 
We introduce a lightweight residual branch parameterized by a single linear mapping layer. Let $X_{flat} \in \mathbb{R}^{B \times (L \cdot M)}$ be the flattened input sequence, and $P \in \mathbb{R}^{12}$ be the semantic prompt matrix. We define the bypass prediction $Y_{bypass}$ as:
$$ Y_{bypass} = W_{bypass} [X_{flat} \oplus P] + b_{bypass} $$
Where $W_{bypass} \in \mathbb{R}^{((L \cdot M) + 12) \times (H \cdot M)}$ maps directly to the forecast horizon $H$. The final prediction linearly fuses the foundation LLM's decoding and our explicit bypass: 
$$ Y_{final} = Y_{LLM} + Y_{bypass} $$

**Zero-Initialization Strategy guarantees Non-Regression:** The weights $W_{bypass}$ and biases $b_{bypass}$ of the statistical bypass are initialized precisely to zero ($W \sim 0, b \sim 0$). Thus, at Epoch 1, Time-VLM+ maps identical loss topologies to the unmodified TimeLLM base, strictly preserving the heavily optimized pre-trained parameters. Over subsequent epochs, gradient descent selectively exploits the simplified bypass mappings, bypassing the LLM's complexity for purely linear relationships, guaranteeing an optimized minima.

## 3.3. Semantic Feature Extraction Platform
Prior to tensor fusion, the raw sequence data maps into a standardized space using $X' = (X - \mu) / (\sigma + \epsilon)$ to prevent prompt magnitude explosion during training. We then extract $P$, capturing structural traits that Transformers traditionally struggle to evaluate over short scales:
1.  **Basic Statistics (5):** Mean ($\mu$), Median, Standard Deviation ($\sigma$), Minimum, Maximum.
2.  **Trend (2):** Global slope scalar ($\frac{X_L - X_0}{L}$) and an absolute trend magnitude derivative.
3.  **Volatility (2):** Local rolling-window variance across stride lengths, contrasted against the global sequence variance.
4.  **Periodicity (1):** Autocorrelation lag-1 estimation $\text{corr}(X_t, X_{t-1})$ mapping high-frequency oscillation severity.
5.  **Stability & Stationarity Shifts (2):** Structural breaks defined by the absolute mean difference and variance difference between sequence subsets: $| \mu(X_{0:L/2}) - \mu(X_{L/2:L}) |$.

## 3.4. Experimental Design
All models are evaluated on multivariate forecasting tasks using identical context windows of $L=96$ to predict horizon $H=96$. The evaluation is tracked via Mean Squared Error (MSE) and Mean Absolute Error (MAE), optimized using Adam with a learning rate of $0.001$ for the prompt-sensitive Time-VLM+ architecture, and $0.005$ for standard baselines.

*   **Experiment 1: Full-Data Forecasting:** All baseline frameworks are challenged on 100% data access environments encompassing the ETTh1, ETTh2, and Weather datasets to establish peak convergence topologies in unconstrained domains.
*   **Experiment 2: Few-Shot Generality:** Models are drastically restricted to 10% uniformly sampled sequence subsets. This isolates whether explicit foundation priors and the structural textual prompting of Time-VLM+ natively overcome catastrophic data deficiency.
*   **Experiment 3: Time-VLM+ Ablation Analysis:** Evaluated consistently at 10% volume, variants of the explicit Time-VLM+ extractors are stripped (`no_periodicity`, `no_trend_stability`, `basic_only` vectors set to null $0.0$-tensors). This maps distinct performance variations to individual temporal features within the textual reprogramming space.
*   **Experiment 4: Computational Efficiency Footprint:** A thorough logging of active Trainable Parameters, cumulative GPU parameter overheads, average training iteration throughputs, and inference latencies maps algorithmic efficiency offsets.
