# 小波特征增强（WFE）替代 Self-Attention 的代码修改方案

## 一、总体目标与策略

- **目标**：用 Haar 小波特征增强（WFE）替代当前扩散模块中的 self-attention 模态融合，在**加强模态融合与模态互信息**的前提下，通过**有序伪序列 + 可学习高频增强 + 残差**设计，尽量**不降低样本多样性**。
- **涉及文件**：仅 `src/models/diffusion_ver15.py`。
- **策略**：
  1. **步骤 A**：调整拼接顺序为 [Vision, Text, Noisy_ID, Time_Emb]，使 Haar 相邻差对应“模态语义差”和“状态/时间差”。
  2. **步骤 B**：新增 WFE 模块（DWT → 可学习增强 D → IDWT → 输出），替换 `selfAttention` 的调用；输出接口保持 `[B, d]`，并加可选残差以保护多样性。
  3. **步骤 C**：移除或保留不用的 self-attention 相关参数（w_q, w_k, w_v, ln），按需保留 `init` 等工具函数。

以下按**修改点**逐条列出位置、修改内容与原因，便于你逐项评判。

---

## 二、修改清单与逐项说明

### 修改 1：`__init__` 中新增 WFE 相关参数，并保留/移除 self-attention 参数

**文件**：`src/models/diffusion_ver15.py`  
**位置**：`diffusion.__init__`，约第 126–137 行（Multimodal Fusion 注释块）。

**当前代码**：
```python
        # Multimodal Fusion:
        self.w_q = nn.Linear(self.embedding_dim,
                         self.embedding_dim, bias=False)
        init(self.w_q)
        self.w_k = nn.Linear(self.embedding_dim,
                         self.embedding_dim, bias=False)
        init(self.w_k)
        self.w_v = nn.Linear(self.embedding_dim,
                         self.embedding_dim, bias=False)
        init(self.w_v)
        self.ln = nn.LayerNorm(self.embedding_dim, elementwise_affine=False)
```

**拟改为**：
- **删除**上述 `w_q, w_k, w_v, ln` 的创建与初始化（不再使用 self-attention）。
- **新增**：
  - `self.wfe_scale = nn.Parameter(torch.ones(2, self.embedding_dim))`  
    用于对两路高频系数 D 做逐维可学习缩放，形状 `[2, d]` 对应两对 (Vis–Text, ID–Time)。初始化为 1，避免一开始压制高频，有利于融合且不损多样性。
  - （可选）`self.ln = nn.LayerNorm(self.embedding_dim, elementwise_affine=False)`  
    若在 WFE 前做一次归一化，可保留；否则可一并删除。

**说明**：  
用“可学习标量/向量”替代 Q/K/V，参数量更少、结构更可控；`[2, d]` 让模型对两路高频（模态差、状态差）分别学习权重，便于加强模态互信息并保留差异以维持多样性。

---

### 修改 2：新增 `wfe(self, X)` 方法，实现 DWT → Enhance → IDWT → 输出

**文件**：`src/models/diffusion_ver15.py`  
**位置**：在 `selfAttention` 方法之前或之后新增一个方法（建议紧接在 `get_timestep_embedding` 之后，约第 281 行后）。

**输入**：`X`，shape `[B, 4, d]`，顺序约定为 [Vision, Text, Noisy_ID, Time_Emb]（由调用处保证）。

**逻辑**：
1. **可选 LayerNorm**：若保留 `self.ln`，则 `X = self.ln(X)`。
2. **Haar DWT**（固定运算，无参数）：
   - 低频：`A0 = (X[:,0] + X[:,1]) / 2`，`A1 = (X[:,2] + X[:,3]) / 2`，堆叠为 `A`，shape `[B, 2, d]`。
   - 高频：`D0 = (X[:,0] - X[:,1]) / 2`，`D1 = (X[:,2] - X[:,3]) / 2`，堆叠为 `D`，shape `[B, 2, d]`。
3. **增强**：`D_tilde = D * self.wfe_scale`（广播），学习对两路高频的加权。
4. **IDWT 重构**：
   - IDWT：  
     `X_recon[:,0] = A[:,0] + D_tilde[:,0]`，`X_recon[:,1] = A[:,0] - D_tilde[:,0]`，  
     `X_recon[:,2] = A[:,1] + D_tilde[:,1]`，`X_recon[:,3] = A[:,1] - D_tilde[:,1]`，  
     得到 `X_recon` shape `[B, 4, d]`。
5. **输出（二选一，建议先采用方案 a）**：
   - **方案 a（推荐）**：`y = X_recon.mean(dim=1)` → `[B, d]`，与当前 self-attention 的 mean 聚合一致，便于公平对比。
   - **方案 b**：`y = X_recon[:, 2, :]`，强调 Noisy_ID 通道，利于保留多样性；若后续希望更突出“去噪目标”，可切换到此方案。
6. **可选残差（保护多样性）**：  
   `y = alpha * y + (1 - alpha) * X[:, 2, :]`，其中 `alpha` 可为 `nn.Parameter(torch.tensor(0.7))` 或固定 1.0（不残差）。  
   若你希望首版尽量少改动，可先不加重残差，仅做 WFE 替换。

**说明**：  
DWT/IDWT 与公式一致；可学习 `wfe_scale` 负责“加强模态融合与互信息”；mean 或取 index=2 控制是否更偏多样性；残差项显式保留 x_noisy 信息，避免融合过度平滑。

---

### 修改 3：`p_losses` 中调整拼接顺序并改为调用 WFE

**文件**：`src/models/diffusion_ver15.py`  
**位置**：约第 171–172 行。

**当前代码**：
```python
        diff_i = torch.cat([x_noisy.unsqueeze(1), t_start.unsqueeze(1), v_start.unsqueeze(1), t_emb.unsqueeze(1)], dim=1)
        predicted_x = self.selfAttention(diff_i)
```

**拟改为**：
```python
        # 有序伪序列: [Vision, Text, Noisy_ID, Time_Emb]，便于 Haar 相邻差对应 (Vis-Text) 与 (ID-Time)
        diff_i = torch.cat([v_start.unsqueeze(1), t_start.unsqueeze(1), x_noisy.unsqueeze(1), t_emb.unsqueeze(1)], dim=1)
        predicted_x = self.wfe(diff_i)
```

**说明**：  
顺序改为 [V, T, ID, Time] 后，DWT 的 (0,1) 对为模态差、(2,3) 对为状态/时间差，与你的物理意义一致；训练时 loss 仍对 `x_start` 与 `predicted_x` 计算，接口不变。

---

### 修改 4：`p_sample` 中同样调整拼接顺序并改为调用 WFE

**文件**：`src/models/diffusion_ver15.py`  
**位置**：约第 204–206 行。

**当前代码**：
```python
        diff_i = torch.cat([x_t.unsqueeze(1), t_t.unsqueeze(1), v_t.unsqueeze(1), t_emb.unsqueeze(1)], dim=1)
        x_start = self.selfAttention(diff_i)
```

**拟改为**：
```python
        # 与 p_losses 一致的顺序: [Vision, Text, Noisy_ID, Time_Emb]
        diff_i = torch.cat([v_t.unsqueeze(1), t_t.unsqueeze(1), x_t.unsqueeze(1), t_emb.unsqueeze(1)], dim=1)
        x_start = self.wfe(diff_i)
```

**说明**：  
推理/采样阶段与训练阶段使用同一顺序和同一融合模块，保证一致性；`x_start` 仍参与后续 `model_mean_x` 等计算，无需改其它逻辑。

---

### 修改 5：删除或保留 `selfAttention` 方法

**文件**：`src/models/diffusion_ver15.py`  
**位置**：约第 283–300 行。

**建议**：  
在确认 WFE 稳定运行后，**删除** `selfAttention` 方法整体，避免遗留未使用代码。若你希望保留“可切换回 attention”的选项，可暂时注释或通过配置分支保留（不推荐长期保留两套实现）。

**说明**：  
删除可减少阅读与维护成本；若需做消融（attention vs WFE），建议用 git 分支或配置项切换实现，而不是在同一文件中保留两套。

---

## 三、不改动的部分（避免误改）

- **`q_sample`、`get_timestep_embedding`、`predict_noise_from_start`**：不改。
- **`p_losses` 的 loss 计算与返回值**：仍为 `loss, predicted_x`，仅 `predicted_x` 的生成方式从 `selfAttention` 改为 `wfe`。
- **`p_sample` 的 `model_mean_x`、后验方差与采样**：不变，仍用 `x_start` 与 `x_t` 计算。
- **`sample` 循环及返回值**：不变。
- **调用方（如 `ccdrec.py` 中的 `p_losses` / `sample`）**：无需改，因接口（输入输出 shape 与语义）保持不变。

---

## 四、可选设计（由你决定是否采用）

| 选项 | 说明 |
|------|------|
| WFE 前是否保留 LayerNorm | **建议首版保留**。作用与原因见下方「LayerNorm 的作用与首版是否保留」。 |
| 输出用 mean 还是取 index=2 | 上文建议先 mean，后续若要强调多样性可改为 `X_recon[:, 2, :]`。 |
| 是否加重残差 `y = alpha * wfe(X) + (1-alpha) * X[:,2,:]` | 建议首版 alpha=1（不残差），若发现多样性下降再加可学习 alpha。详见下方**五、如何检查多样性是否下降**。 |
| `wfe_scale` 形状 | `[2, d]` 已足够；若想更省参数可用 `[2, 1]` 或 `[1, d]`。 |

**LayerNorm 的作用与首版是否保留**  
- **作用**：对 `X` 在序列维（4 个模态）上做归一化，使每个特征维在 4 个 token 上均值为 0、方差为 1（`elementwise_affine=False` 时无额外参数）。这样四路输入在进入 DWT 前量纲一致，做差时不会被某一模态尺度主导，高频 D 更稳定，且与原 self-attention 前有 `ln` 的设定一致，对比更公平。  
- **首版是否保留**：**建议首版保留**。与 baseline 一致、利于训练稳定、且无额外参数；后续可做消融「去掉 ln」再决定长期是否保留。

---

## 五、如何检查多样性是否下降

在改用 WFE 后，若要判断“样本多样性”是否下降，可从**推荐结果多样性**和（可选）**嵌入/采样多样性**两方面做对比（WFE 版 vs 原 self-attention 版，同一数据集、同一划分、可比训练设置）。

### 1. 推荐结果层面的多样性（优先做）

在现有评估流程里，已有**每个用户**的 top-K 推荐列表 `topk_index`（例如在 `TopKEvaluator.evaluate` 中 `torch.cat(batch_matrix_list, dim=0).cpu().numpy()`）。在此基础上可加算以下指标，**在验证/测试集上按 epoch 或最终结果记录**，与 baseline（原 attention）对比：

| 指标 | 含义 | 计算方式 | 若多样性下降 |
|------|------|----------|----------------|
| **Coverage** | 被推荐过的物品种类占比 | 全体用户 top-K 并集去重后，不同 item 数 / 全量 item 数 | 覆盖率明显降低 |
| **Gini（推荐次数分布）** | 推荐量在不同 item 上的集中程度 | 对“每个 item 被推荐次数”算 Gini 系数，越大越集中 | Gini 升高表示更集中、多样性变差 |
| **Entropy（推荐次数分布）** | 同上，用熵表示 | 对“每个 item 被推荐次数”做归一化后算熵 | 熵降低表示更集中、多样性变差 |

- **实现位置建议**：在 `src/utils/topk_evaluator.py` 的 `evaluate` 中，在得到 `topk_index`（shape 约 `[num_users, max_k]`）之后：
  - 统计所有出现过的 item id 的集合，除以全量 item 数 → **Coverage**；
  - 统计每个 item 被推荐次数（可只在对应用户的 top@K 内计数），再算 **Gini** 或 **Entropy**。
- **对比方式**：同一数据集上，用原模型和 WFE 模型各跑一份验证/测试，看同一 K（如 20）下 Coverage 是否明显变低、Gini 是否明显变高（或 Entropy 变低）。若变化不大或略好，可认为推荐多样性未下降。

### 2. 嵌入/采样层面的多样性（可选、更诊断用）

若担心扩散**采样过程**本身变得过于确定、导致表示趋同，可做：

- **同一输入多次采样的方差**：对同一批 item（如固定 `x_start, t_start, v_start`），多次调用 `diff.sample()`（每次前向的噪声不同），得到多份 `sample_x`；对同一 item 的嵌入在多次采样下的方差（或不同 item 嵌入的协方差矩阵的迹）做统计。WFE 若过度平滑，同一 item 多次采样的方差可能变小。
- **跨 item 嵌入的离散度**：在单次 `sample()` 后，对所有 item 的 `sample_x` 做 PCA 或直接算各维度方差均值；若 WFE 版明显低于 baseline，说明嵌入更“挤在一起”，可作为多样性下降的参考信号。

这两类指标不需要改训练流程，可在**固定模型权重**下写小脚本对验证集或子集做几次采样即可。

### 3. 建议的落地顺序

1. **先加 Coverage**：在 `TopKEvaluator.evaluate` 里增加 Coverage 计算并写入返回的 `metric_dict`（如 `'Coverage@20'`），训练时与 Recall/NDCG 一起打日志；改 WFE 前后各跑一次，对比 Coverage。
2. 若需更细判断，再加 **Gini 或 Entropy**（同样在 `evaluate` 里，基于同一 `topk_index`）。
3. 若仍怀疑是“采样多样性”问题，再做 **嵌入方差/离散度** 的离线脚本。

这样即可在“是否加重残差”的决策上有依据：若发现 WFE 版 Coverage 明显跌、Gini 升，再考虑加可学习 alpha 残差或改用“取 index=2”等设计。

---

## 六、小结：修改项与目的一览

| 序号 | 位置 | 修改内容 | 目的 |
|------|------|----------|------|
| 1 | `__init__` | 删 w_q/w_k/w_v/ln，新增 wfe_scale（及可选 ln） | 用 WFE 参数替代 attention 参数，可学习两路高频权重 |
| 2 | 新增方法 | 实现 `wfe(X)`：DWT → D*scale → IDWT → mean/取位 → 可选残差 | 完成小波融合与可学习增强，输出 [B,d] 接口不变 |
| 3 | `p_losses` | 拼接顺序改为 [v,t,x_noisy,t_emb]，`selfAttention`→`wfe` | 有序伪序列 + 用 WFE 做训练时融合 |
| 4 | `p_sample` | 同上顺序与 `wfe` 调用 | 与训练一致，采样阶段也用 WFE |
| 5 | 方法删除 | 删除 `selfAttention` | 避免死代码，保持清晰 |

若你认可上述每一项（或对可选项做出选择），再按此方案在 `diffusion_ver15.py` 中实施即可；若有某一条希望保留现状或换做法，指出序号即可，我可以按你的选择给出精确 diff 或代码片段。
