# RSNA Knee — `r224` 多尺度 Stage-1 修复报告（2026-10-13 23:18 PDT / 10-14 06:18 UTC）

> 起因：一份转发分析主张失败源于「CoAtNet 相对位置偏置表与网格绑定」，并给出「三行 GPU interpolate」修法。
> **结论：诊断错误、修法不对；真因是 `raptor_run.py` 里三处硬编码分辨率断言。已全部修复并推 v2。**
> **⚠ 修订说明**：本报告初版只找到 1 处守卫；v1 推送后仍 3/3 FALLBACK，实测证明有 **3 处**。本版已更正。

---

## 一、真因：三处硬编码分辨率断言（不是 pos-bias，也不是一处）

`scratchpad/heliosli/extracted_sources/raptor_run.py`

| # | 位置 | 断言 | 触发条件 |
|---|---|---|---|
| 1 | L294 `eval_windows` | `assert tri.shape[-2:] == (IMG,IMG) and res == INPUT_RES` | 窗口构建期 |
| 2 | L317 `gpu_windows` | 同上 | 窗口构建期（**live 路径**） |
| 3 | **L278 `infer_pair_probs`** | `assert plain.shape==mirrored.shape==(94,3,320,320)` | 前向期；live 路径 **L567** 调用 |

**因果链**：`INPUT_RES=320`；原始 `load_model` 返回 `ck_res=INPUT_RES`（所以**基线 r384 臂实际以 320 运行**）；builder 把 `ck_res` 改成 224 → 三处断言逐一失败（`224==320`、`(94,3,224,224)!=(94,3,320,320)`）→ 每 study `FALLBACK (AssertionError: )`。

**⚠ v1 的教训**：只放宽了 L294/L317 → **仍然 3/3 FALLBACK**。必须**搜全所有硬编码 res 断言**（`grep -nE "assert|320|\(94,"`）再推。

**反驳 pos-bias 论**：若真有分辨率相关偏置表，构建 224 模型时 `strict=True` 的 `load_state_dict` 就会 shape 不匹配而失败——但它成功了（`loaded 2 arm(s)`）。

---

## 二、正确修法（4 处补丁，17 项自检全 PASS）

`scratchpad/build_0957_s1_multiscale.py`：

1. **新增 `infer_res`**：r384 臂钉 `320`（逐字节复现基线）、r224 臂 `224`；
2. `ck_res = int(infer_res) if infer_res else INPUT_RES`，call site 传 `infer_res=arm.get("infer_res")`；
3. **放宽 L294/L317** → `assert tri.shape[-2:] == (IMG, IMG)`；
4. **放宽 L278** → `assert plain.shape==mirrored.shape and plain.ndim==4 and plain.shape[1]==3`。

> ⚠ 只做第 3、4 步而不做第 1 步，r384 臂会从 320 漂到 384，**偏离已验证的 0.956 基线**。

**几何**：res=320 → 中心裁剪 83.3% FOV；res=224 → 58.3% FOV。`lo=(384-224)//2=80` → `tri[:,80:304,80:304]` 已是 224×224 → 紧随的 `F.interpolate` 是**死代码**。即 r224 **保持 mm/px、只收 FOV**（自带 2,399 例 OAI 先验的关节中心紧裁剪专修臂），与 `dinosaur-the-last` 参考核（`ns['INPUT_RES']=224` 单臂）同路径。

---

## 三、⚠ 门禁陷阱（本次靠它拦下 v1）

`blend_receipt.json` 的 **`raptor_failed==0` 不能证明没 fallback**——fallback 会填合法值。v1 就是 `raptor_failed==0` 但日志 3/3 FALLBACK。
**必须**在日志里 grep 字面量 **`FALLBACK`**。守护 `scratchpad/watch_ms384_224.py` 的 9 项门禁因此拒绝了 v1 提交（**未烧配额**）。

---

## 四、状态

- **v2 已推**：`easonyanyan/rsna-knee-0957-multi-scale-stage-1`（**RUNNING**）。
- 守护 `scratchpad/watch_ms384_224.py`（task `GwJGAx`）等 COMPLETE → 9 项门禁 → 自动提交 v2。
- 配额：easonyanyan **2/5**（v1 被拦下，未消耗）。

## 五、下一步
过门禁后自动提交（第 3/5）。若 >0.956 → 跨尺度是首个公开侧真增益，固化进主攻核；若 =0.956 → Stage-1 第二臂轴彻底关闭。
