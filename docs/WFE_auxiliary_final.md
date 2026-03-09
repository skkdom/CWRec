# 小波特征增强（WFE）辅助分支方案说明（最终有提升版本）

## 一、总体目标与策略

- **目标**：在**不替换**主干 self-attention 的前提下，将 WFE 作为**小残差辅助分支**注入扩散模块的模态融合，以加强模态对齐与细粒度语义利用，同时保持推荐性能不降、甚至微超 baseline。
- **涉及文件**：`src/models/diffusion_ver15.py`、`src/configs/model/ccdrec.yaml`。
- **策略**：
  1. **主干保留**：self-attention 仍作为主融合模块，输入顺序保持 `[ID, Text, Vision, Time]`。
  2. **WFE 作为辅助**：在主干输出基础上，加入 `gate * (x_wfe - x_main)` 的小残差校正，其中 `gate` 为可学习门控（带 `eps_max` 上限）。
  3. **有序伪序列**：WFE 分支输入采用 `[Vision, Text, Noisy_ID, Time_Emb]`，使 Haar 相邻差对应 (V−T) 模态差 与 (ID−Time) 状态差。
  4. **实验开关**：通过 `wfe_ablation`、`wfe_in_sample` 支持消融与采样阶段关闭 WFE 的对比实验。

以下先给出**数学原理与公式**，再按**修改点**逐条列出位置、修改内容与原因。

---

## 二、数学原理与公式

### 2.1 有序伪序列

设输入四路模态为 Vision $V$、Text $T$、Noisy_ID $X_{\text{noisy}}$、Time Embedding $E_t$，构造有序伪序列：

$$
S = [V,\; T,\; X_{\text{noisy}},\; E_t] \in \mathbb{R}^{B \times 4 \times d}
$$

**设计动机**：将“内容模态”与“状态信息”分组，使 Haar 小波的相邻差分具有明确物理意义：

- **Pair 1** $(V, T)$：$D_1 = (V - T)/2$ 表示模态间语义差异（互补信息）
- **Pair 2** $(X_{\text{noisy}}, E_t)$：$D_2 = (X_{\text{noisy}} - E_t)/2$ 表示当前状态与噪声水平的差异（即需预测的噪声残差方向）

### 2.2 Haar 小波分解（DWT）

对长度 4 的序列，Haar 单层分解得到 2 个低频系数 $A$ 与 2 个高频系数 $D$：

$$
A_i = \frac{S_{2i} + S_{2i+1}}{2}, \quad
D_i = \frac{S_{2i} - S_{2i+1}}{2}, \quad i \in \{0, 1\}
$$

即：

$$
\begin{aligned}
A_0 &= \frac{V + T}{2}, &
A_1 &= \frac{X_{\text{noisy}} + E_t}{2}, \\
D_0 &= \frac{V - T}{2}, &
D_1 &= \frac{X_{\text{noisy}} - E_t}{2}
\end{aligned}
$$

### 2.3 高频增强

引入可学习缩放矩阵 $\mathbf{T} \in \mathbb{R}^{2 \times d}$（对应 `wfe_scale`），对高频系数逐元素加权：

$$
\tilde{D}_i = D_i \odot \mathbf{T}_i
$$

其中 $\odot$ 表示逐元素乘法。模型据此学习如何根据模态差和状态差加权高频信息。

### 2.4 小波重构（IDWT）

由 $A$ 与 $\tilde{D}$ 还原长度为 4 的序列：

$$
\hat{S}_{2i} = A_i + \tilde{D}_i,\quad \hat{S}_{2i+1} = A_i - \tilde{D}_i
$$

即：

$$
\begin{aligned}
\hat{S}_0 &= A_0 + \tilde{D}_0, &
\hat{S}_1 &= A_0 - \tilde{D}_0, \\
\hat{S}_2 &= A_1 + \tilde{D}_1, &
\hat{S}_3 &= A_1 - \tilde{D}_1
\end{aligned}
$$

**WFE 输出**：取去噪目标对应位置，$x_{\text{wfe}} = \hat{S}_2 \in \mathbb{R}^{B \times d}$。

### 2.5 主干与辅助残差融合

设主干 self-attention 输出为 $x_{\text{main}}$，WFE 输出为 $x_{\text{wfe}}$。最终预测采用可学习门控的残差形式：

$$
\hat{x} = x_{\text{main}} + \gamma \cdot (x_{\text{wfe}} - x_{\text{main}})
$$

其中门控 $\gamma$ 定义为：

$$
\gamma = \epsilon_{\max} \cdot \sigma(g), \quad \sigma(g) = \frac{1}{1 + e^{-g}}
$$

- $\epsilon_{\max}$：门控上限（如 0.15），由配置 `wfe_eps_max` 设定
- $g$：可学习标量 `wfe_gate_logit`，初始化为 -4，使 $\sigma(-4) \approx 0.018$，训练可自动调节

**等价形式**：$ \hat{x} = (1-\gamma) x_{\text{main}} + \gamma x_{\text{wfe}} $，即主干与 WFE 的凸组合；$ \gamma \to 0 $ 时退化为纯主干（baseline）。

---

## 三、修改清单与逐项说明

### 修改 1：`__init__` 中保留 self-attention 并新增 WFE 相关参数

**文件**：`src/models/diffusion_ver15.py`  
**位置**：`diffusion.__init__`，Multimodal Fusion 相关块。

**实现**：
```python
        # Multimodal Fusion:
        # 主干仍采用 self-attention，WFE 作为小残差辅助分支
        self.w_q = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        init(self.w_q)
        self.w_k = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        init(self.w_k)
        self.w_v = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        init(self.w_v)

        # WFE 高频缩放参数
        self.wfe_scale = nn.Parameter(torch.ones(2, self.embedding_dim))
        # WFE 残差门控：gate = eps_max * sigmoid(logit)，默认接近 0，训练自行决定是否打开
        self.wfe_eps_max = config['wfe_eps_max']
        self.wfe_gate_logit = nn.Parameter(torch.tensor(-4.0))
        # 消融/实验开关
        self.wfe_ablation = config['wfe_ablation']   # True: 完全关闭 WFE，等价 baseline
        self.wfe_in_sample = config['wfe_in_sample']  # False: 采样阶段只用主干，不用 WFE
        # 共享的一层 LayerNorm，用于 attention 与 WFE
        self.ln = nn.LayerNorm(self.embedding_dim, elementwise_affine=False)
```

**说明**：  
主干 Q/K/V 不变；`wfe_scale` 对两路高频 (V−T)、(ID−Time) 做可学习缩放；`wfe_gate_logit` 初始化为 -4，使 `sigmoid(-4)≈0.018`，训练可自行决定是否增大 gate；`wfe_eps_max` 控制 gate 上限，实验表明 0.15 时效果最佳（Recall@10、NDCG@10 可微超 baseline 约 0.0001）。

---

### 修改 2：新增 `wfe(self, X)` 方法

**文件**：`src/models/diffusion_ver15.py`  
**位置**：`get_timestep_embedding` 之后。

**逻辑**：
1. **LayerNorm**：`X = self.ln(X)`。
2. **Haar DWT**：  
   - 低频：`A0 = (X[:,0]+X[:,1])/2`，`A1 = (X[:,2]+X[:,3])/2`  
   - 高频：`D0 = (X[:,0]-X[:,1])/2`，`D1 = (X[:,2]-X[:,3])/2`  
3. **增强**：`D_tilde = D * self.wfe_scale`。  
4. **IDWT 重构**：`X_recon` 恢复为 `[B, 4, d]`。  
5. **输出**：`y = X_recon[:, 2, :]`，取 ID 通道（去噪目标）。

**说明**：  
WFE 输出仅作为辅助校正方向，不直接替代主干；取 ID 通道与预测目标一致，利于学习有意义的模态融合残差。

---

### 修改 3：`p_losses` 中主干 + WFE 残差

**文件**：`src/models/diffusion_ver15.py`  
**位置**：`p_losses` 内，构建 `diff_main`、`diff_wfe` 及 `predicted_x` 部分。

**实现**：
```python
        # 主干: 保持原始顺序 [ID, Text, Vision, Time]，用 self-attention 融合
        diff_main = torch.cat(
            [x_noisy.unsqueeze(1), t_start.unsqueeze(1), v_start.unsqueeze(1), t_emb.unsqueeze(1)],
            dim=1
        )
        predicted_x_main = self.selfAttention(diff_main)

        if self.wfe_ablation:
            predicted_x = predicted_x_main  # 消融：完全关闭 WFE，等价 baseline
        else:
            # 辅助分支: 有序伪序列 [Vision, Text, Noisy_ID, Time_Emb]
            diff_wfe = torch.cat(
                [v_start.unsqueeze(1), t_start.unsqueeze(1), x_noisy.unsqueeze(1), t_emb.unsqueeze(1)],
                dim=1
            )
            predicted_x_wfe = self.wfe(diff_wfe)
            gate = self.wfe_eps_max * torch.sigmoid(self.wfe_gate_logit)
            predicted_x = predicted_x_main + gate * (predicted_x_wfe - predicted_x_main)
```

**说明**：  
`wfe_ablation=True` 时等价 baseline；否则主干输出加小残差，gate 由训练自动学习，可趋近 0 或适度增大。

---

### 修改 4：`p_sample` 中同样主干 + WFE 残差，并支持采样阶段关闭

**文件**：`src/models/diffusion_ver15.py`  
**位置**：`p_sample` 内，构建 `x_start_main` 与 `x_start` 部分。

**实现**：
```python
        x_start_main = self.selfAttention(diff_main)

        if self.wfe_ablation or not self.wfe_in_sample:
            x_start = x_start_main  # 消融 / 采样阶段关闭 WFE：只用主干
        else:
            diff_wfe = torch.cat(...)
            x_start_wfe = self.wfe(diff_wfe)
            gate = self.wfe_eps_max * torch.sigmoid(self.wfe_gate_logit)
            x_start = x_start_main + gate * (x_start_wfe - x_start_main)
```

**说明**：  
`wfe_in_sample=False` 时，采样阶段不引入 WFE，仅训练时用 WFE 辅助，可用于验证“采样阶段关闭 WFE”对 test 稳定性的影响。

---

### 修改 5：配置文件新增 WFE 相关项

**文件**：`src/configs/model/ccdrec.yaml`  

**新增**：
```yaml
# WFE 实验开关
wfe_eps_max: 0.15
wfe_ablation: false   # true: 消融，完全关闭 WFE（等价 baseline）
wfe_in_sample: true   # false: 采样阶段关闭 WFE，训练用 WFE、推理只用主干
```

**说明**：  
`wfe_eps_max` 建议 0.15；消融实验时将 `wfe_ablation` 置为 `true`；“采样阶段关闭 WFE”实验时将 `wfe_in_sample` 置为 `false`。

---

## 四、不改动的部分

- **`q_sample`、`get_timestep_embedding`、`predict_noise_from_start`**：不变。
- **`p_losses` 的 loss 计算与返回值**：仍为 `loss, predicted_x`，仅 `predicted_x` 的生成方式为 `main + gate*(wfe - main)`。
- **`p_sample` 的 `model_mean_x`、后验方差与采样**：不变。
- **`sample` 循环及返回值**：不变。
- **调用方（如 `ccdrec.py`）**：无需修改。

---

## 五、实验效果与配置建议

| 配置 | 训练 | 采样 | 说明 |
|------|------|------|------|
| 默认（有提升） | 主干 + WFE | 主干 + WFE | Recall@10、NDCG@10 可微超 baseline |
| 消融 | 仅主干 | 仅主干 | 验证提升来自 WFE |
| 采样关闭 WFE | 主干 + WFE | 仅主干 | 可选：测试采样阶段不用 WFE 的稳定性 |

**推荐超参**：`wfe_eps_max: 0.15`；`wfe_ablation: false`；`wfe_in_sample: true`。

---

## 六、小结：修改项与目的一览

| 序号 | 位置 | 修改内容 | 目的 |
|------|------|----------|------|
| 1 | `__init__` | 保留 w_q/w_k/w_v/ln，新增 wfe_scale、wfe_eps_max、wfe_gate_logit、wfe_ablation、wfe_in_sample | 主干不变，增加 WFE 辅助与实验开关 |
| 2 | 新增方法 | 实现 `wfe(X)`：DWT → D*scale → IDWT → 取 ID 通道 | 提供模态融合残差方向 |
| 3 | `p_losses` | 主干 attention + 条件 WFE 残差，支持 wfe_ablation | 训练时引入可学习 gate 的 WFE 辅助 |
| 4 | `p_sample` | 同上逻辑，支持 wfe_ablation 与 wfe_in_sample | 采样阶段可沿用或关闭 WFE |
| 5 | `ccdrec.yaml` | 新增 wfe_eps_max、wfe_ablation、wfe_in_sample | 便于复现与消融实验 |

此方案在 baby 数据集上可达 Recall@10≈0.0674、NDCG@10 微超 baseline（约 +0.0001），且通过可学习 gate 与实验开关支持进一步消融与调参。
