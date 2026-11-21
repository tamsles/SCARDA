# TASKS.md — 里程碑与验收标准

每个里程碑结束后停下来按 AGENTS.md §3 汇报，等人类确认再继续。日期为计划日期，不代表已完成。

---

## M0 环境与基线复现（11/20）

1. Fork 仓库，新建 `requirements.txt`（Python ≥3.10、torch 2.x、torchvision、numpy、scipy、pyyaml、pandas、matplotlib、pytest）。
2. 在新环境下运行原始 `main.py` 的缩短版（`-ep 2 -run_times 1`），确认原代码可跑。
3. 实现 `scarda/risk.py::scar_ovr_risk`（SPEC §3.1–3.2）。
4. `tests/test_risk.py`：
   - **T1 复现**：单热 $\bar y$、均匀 $\pi$、`pibar_mode=batch`、无权重、`abs` → 与根目录 `utils_algo.SCARCE_loss` 相差：float64 下绝对误差 < 1e-6；另加 float32 用例，相对误差 < 1e-5（多组随机 logits，含某类空组的情形）。
   - **T2 单位权重**：`w_pos=w_neg=1` 与 `None` 结果相同。
   - **T3 修正分支**：构造 $\hat P_k<0$ 的输入，`abs`/`relu`/`none` 三种结果符合定义，`neg_rate` 正确。

**验收**：原 `main.py` 跑通的日志；`pytest tests/test_risk.py` 全部通过。

**M0 审查遗留（已知、暂不处理）**：
- T1 float64 的 1e-6 容差余量有限：`SCARCE_loss` 用 float32 构造先验（`utils_algo.py` L34–36，`ccp` 与 0.9 差约 2.4e-8），测试传入精确 float64 `1/K`。随机对称 logits 下实测差 9e-8–5.9e-7；非对称 logits（±10）下约 2.4e-6，会超过 1e-6。改用逐位相同的先验后差 ≤ 4e-15。判断为参照实现的精度问题而非 `risk.py` 的 bug，为可读性不在测试中处理。若今后 T1 在新用例上失败，先排查此原因。
- `scar_ovr_risk` 不检查权重中的 NaN/inf：NaN 能通过 `w < 0` 的校验；组外的 inf 乘以掩码 0 得 NaN。两者都会使 `total` 变为 NaN 且不报错。SPEC 未要求此校验，留到 M3 实现 DIW 时根据权重来源决定是否处理。
- `scarda/risk.py::batch_pibar` 未被调用（死代码），对整数 ybar 固定返回 float32，与主路径跟随 logits dtype 不一致。〔已解决：M2 后重构中删除〕

---

## M1 数据协议 + source_only 端到端（11/20–21）

1. `scarda/data.py`：按 SPEC §4 构造四个池、偏移先验、互补标签、`WeakSet` / `LabelVault`、split 哈希。〔重构注：`LabelVault` 已改为具名元组 `Labels`，split 哈希已移除〕
2. `scarda/models.py`、`scarda/train.py`（先只支持 `source_only`、`target_only`）、`scarda/evaluate.py`、`scarda/io_utils.py`、`scripts/run.py`。
3. `tests/test_data.py`（**T4**）：各池类计数精确等于目标计数；池两两不相交；$\bar y$ 从不包含真类；经验 $c$ 与设定值的偏差在 3 个标准误内（按池合并估计：$\hat c$ = 该池所有 $k\neq y$ 的 $(i,k)$ 对上 $\bar y_{ik}$ 的均值，$N$ = 这类对的个数，$\mathrm{SE}=\sqrt{c(1-c)/N}$；不逐类检验）；同 seed 两次生成哈希一致。〔重构注：已改为直接比较各池索引与 $\bar y$ 数组〕
4. `scripts/bench_time.py`：测 `source_only` 每 epoch 耗时（以及 GPU/CPU 信息）。
5. 在真实 MNIST 上跑 `source_only`，$\rho=20$，seed 0，完整 epoch 数。

**验收**：`results/runs/` 下有一个完整 run 的 4 个文件；打印出的各池实际先验与 $\hat{\bar\pi}$ 与设定一致；给出单 epoch 耗时，据此提出 `E` 的冻结值与矩阵规模建议（不自行决定）。

**M1 结果记录**：
- 状态：子任务 1–5 已完成（commits `da6e207`, `6ff9d21`, `f92510e`, `5c6ba80`）；`pytest -q` 73 passed。
- 切分、隔离、哈希、seed 的确认方案已写入 SPEC §4.1/§4.3/§4.4/§4.5〔重构注：§4.4 已改写为重构后的结构，哈希定义已移除〕；训练与输出约定写入 SPEC §7/§8。
- 本地 run：`results/runs/source_only_rho20_nte2000_seed0_9bb48ece/`（RTX 3070 Laptop，100 epoch，0.307 s/epoch，git dirty），仅作开发验证，不计入主矩阵。
- 决定：$E=100$（暂定，M3 冻结时最终确认）；矩阵规模在 M3 测得 Wisteria 上的 DIW 耗时后按 SPEC §9.2 的规则决定。计算平台为 Wisteria/BDEC-01（SPEC §9.1）。
- 小重叠：`target_only` 的训练代码已在 M1 实现，M2 只需跑 run。

---

## M2 Oracle 权重 + 风险恒等式检查 + target_only（11/21）

1. `scarda/oracle.py`（SPEC §5），`train.py` 支持 `oracle_iw` 与 `target_only`。
2. `tests/test_oracle.py`（**T5**）：在合成 3 类高斯混合 + 类先验偏移数据上，大样本下每个 $(k,s)$ 组 oracle 权重均值 ≈ 1。
3. `tests/test_risk.py` 增加 **T6 恒等式（合成数据）**：固定线性分类器，`correction=none`，以下三者在重复抽样均值上相差不超过 3 个标准误：
   (a) 带真标签的测试域 OVR 风险；(b) 测试域弱标签 SCAR 风险（真 $\pi_{te}$）；(c) oracle 加权训练域风险。
4. `scripts/check_identity.py`：在 MNIST 上对两个固定分类器（随机初始化、source_only 训练 1 epoch 后）计算同样三者，输出到 `results/checks/identity.json`。
5. 跑 `oracle_iw`、`target_only` 各 1 个 run（$\rho=20$，$n_{te}=2000$，seed 0）。

**验收**：T5、T6 通过；`identity.json` 中三者一致（差异在给出的标准误范围内）；两个 run 完整落盘。
**止损**：恒等式不一致 → 停止，不进入 M3，上报推导与数值。若 oracle_iw 的 accuracy 不优于 source_only，不视为失败，照常记录。

**M2 结果记录**：
- 状态：子任务 1–5 已完成（commits `ca14288`, `5bbac3d`, `48cb74d`, `a9c674c`, `a6dd4b5`；审查后修正 `1674a38`, `40b790a`）；`pytest -q` 95 passed。
- 独立审查结论：`oracle.py` 与 SPEC §5 逐项一致；§5 公式经纸面推导成立（只用到每个 $k$ 的边缘 SCAR，`single_uniform` 同样适用）；T6 三种风险均为 `correction=none`、同一固定分类器；oracle 只读 `S_tr` 真标签；bootstrap 标准误计算本身正确。
- 审查发现并已修正：`source_only_1epoch` 用 `S_tr.ybar` 训练，原先 (c) 也在同一份 `S_tr.ybar` 上计算，偏低约 0.20（对 40 次独立重抽 $\bar y$，0/40 低于训练标签上的值）。修正：(c) 与 control 改在 `S_tr` 互补标签的**独立重抽**上计算（`oracle.fresh_source_pool`〔重构注：现名 `oracle.fresh_source_ybar`〕，`SeedSequence([seed, 101])`，同 `gen` 与 $c_{tr}$，保留 $x,y$）；训练标签上的值作为诊断写入 `in_sample_training_ybar`。这是对 M2.4 检查方法的补充，分类器定义未变。
- `results/checks/identity.json`（commit `1674a38`，clean；$\rho=20$, seed 0, B=2000）：

  | 分类器 | a−b | a−c | b−c | a−control |
  |---|---|---|---|---|
  | `random_init` | z +1.52 | z −0.31 | z −1.59 | z −2.75（未检出） |
  | `source_only_1epoch` | z +0.12 | z −0.53 | z −0.30 | z −5.80（检出） |

- Runs（commit `a6dd4b5`，clean；修正只新增函数、不影响训练路径）：`oracle_iw_rho20_nte2000_seed0_5e7c20bf`（acc_last10 0.8928）、`target_only_rho20_nte2000_seed0_201c1b18`（acc_last10 0.5995）。

**M2 审查遗留（已知、暂不处理）**：
- **MNIST 恒等式检查检出力不足**：`random_init` 的阴性对照 a−control 未越过 3 SE（z −2.75）；把 oracle 权重全换为 1（仍用 $\pi_{te}$、T_tr 的 $\hat{\bar\pi}$）在修正前的数据上 a−c 的 z 为 +2.49 / +1.41，同样会判为一致。即 `identity.json` 对"权重是否正确"几乎没有区分力，权重正确性的证据主要来自 T5/T6。可选改进：增加"只去掉权重"的对照并在 JSON 中记录是否检出。
- **(b) 精度过低**：T_val 仅 2000，`source_only_1epoch` 上 (b) 的 SE ≈ 0.17（风险值 1.43），a−b、b−c 几乎无信息。
- **(a) 与 (b)(c) 来自不同基数据**：(a) 用 MNIST test（T_eval），(b)(c) 用 MNIST train。恒等式隐含两者 $p(x\mid y)$ 相同。真标签、按 $\pi_{te}$ 重加权的 S_tr 风险与 (a) 之差 z = +2.82（random_init）/ −2.27（source_only_1epoch），方向相反、处于临界附近，不足以下结论；报告中须声明此假设。
- **`group_weight_tables` 不校验 $c\in[0,1)$**：$c_{tr}=1$ 时 `W_neg` 出现 inf/NaN，乘掩码 0 后得 NaN（即 M0 遗留的"组外 inf × 0"路径）。默认配置不触发。
- **T6 阴性对照不隔离权重**：`test_T6_negative_control_detected` 的 control 同时换了 $\pi$、$\bar\pi$ 并去掉权重，测的是"偏移存在"。T6 对权重错误本身是敏感的（审查中的变异测试：权重全为 1 → |z| ≈ 47–49；$w^-$ 去掉 $c$ 因子 → z +29.8 / +3.6；仅 $w^+$ 置 1 → prototype 检不出（z −0.6），random z −147），但 $w^-$ 的 $c$ 因子错误在 random 分类器上只有 z +3.6，余量小。

**M2 后重构（2026-11-21，按 AGENTS §1 精简）**：
- 移除 `LabelVault`（改为具名元组 `Labels`）、`SplitRecord`/`PoolSpec`、split 哈希与 `base_hash`、git dirty 记录、`train()`/`fit()` 重复入口、`risk.batch_pibar` 死代码（即 M0 遗留第 3 条）；`oracle_weights_from_vault`/`fresh_source_pool` 改名为 `source_oracle_weights`/`fresh_source_ybar`，直接接收 `y`；新增 `models.build_model`/`predict_logits` 收拢重复代码。
- 不变：`SeedSequence` tag、切池顺序、`gen_ybar`、`scar_ovr_risk`、batch 顺序、配置哈希（三个已有 run 的目录名不变）。
- 验证（CPU）：重构前后在 MNIST（$\rho\in\{20,5\}$）上的各池 $x$/$\bar y$/索引/真标签、oracle 权重、独立重抽、三种方法 2 个 epoch 的全部指标与模型参数，共 33 项逐位相同；`check_identity.py`（B=100）的 `classifiers` 块完全相同。`pytest -q` 94 passed（删去 1 个隔离结构测试）。M1/M2 的已记录结果无需重跑。

---

## M3 DIW + 冻结（11/22）

1. `scarda/diw.py`（SPEC §6），`train.py` 支持 `diw`、`diw_combined`，记录全部诊断字段。
2. `tests/test_diw.py`（**T7**）：
   - 求解器与 `scipy.optimize.minimize`（SLSQP 或 trust-constr）在随机小问题上目标值相对差 < 1e-4，且约束满足；
   - $G_{te}$ 与 $G_{tr}$ 是同一组样本时，权重 ≈ 1；
   - 空组 / 小组触发 fallback 且计数正确。
3. 按 SPEC §7 对 $B\in\{5,10\}$ 做选择，写入 `configs/frozen.yaml`；**在 Wisteria 上**用 `scripts/bench_time.py` 测 `source_only` 与 DIW 的单 epoch 耗时，按 SPEC §9.2 的规则给出矩阵规模建议（$P$、$H$ 与最终方案由人类确认）。

（2026-11-21 精简：原 T8 标签隔离结构测试、T9 smoke 测试、CI 已移除，见 AGENTS §1。）

**验收**：全部测试通过；`frozen.yaml` 提交；人类确认冻结方法、数据协议、选择规则与矩阵（SPEC §9）后，M3 结束。冻结后只允许修 bug，不加功能。
**止损**：若 DIW 训练出现 NaN / 权重退化（ESS 比例长期 < 0.05）且当天无法定位，上报；主结果改为 source/target/oracle 三者，DIW 作为部分结果与失败分析。

---

## M4 主实验（11/23–24）

1. `scripts/run_matrix.py` 读取 `configs/matrix_main.yaml`，跳过已有 `final.json` 的 run（并行交给作业调度器）。
2. 在 Wisteria/BDEC-01 上搭建环境并提交作业（作业脚本的位置与形式待定，开始 M4 前与人类确认）。
3. 跑 M3 冻结的矩阵（SPEC §9）+ $\rho=1$ 对照，全部 run 在同一平台完成。

**验收**：每个 run 都有 `final.json`；失败的 run 有日志说明原因；不得因结果不理想而删除或重跑某个 seed。

---

## M5 汇总与分析材料（11/25）

1. `scripts/summarize.py`（SPEC §8 输出）：每个方法 × 条件的均值 ± 标准差、配对差及 95% 区间、ESS 与 fallback 统计、修正触发率。

**验收**：`results/summary/` 下有 SPEC §8 列出的表与图；报告中的数字都能在 `results/` 下找到（AGENTS §2.5）。

---

## M6 说明文档（11/26–27）

1. 写 `README_SCARDA.md`：设定、与原 SCARCE 的差异、安装、单个实验命令、矩阵命令、汇总命令、已知局限（oracle prior、仅类先验偏移、单数据集、自归一化有偏、ABS 修正沿用官方代码）。

**验收**：`README_SCARDA.md` 提交。（原"全新环境复现 + `clean_env_log.txt`"已移除。）
