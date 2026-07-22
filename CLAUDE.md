# Open Duck — Workspace 總覽

這個目錄是一個 **workspace（並排放置的多 repo 工作區）**，不是單一 git repo。
底下四個子目錄各自是獨立的 git repository（各有自己的 `.git` 與 GitHub remote），
`open_duck_all/` 本身只負責把它們擺在一起，並記錄它們之間的關係。

專案本體是 [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini)——一台約 42cm 高、
BOM 成本低於 $400 的迷你 BDX 雙足機器人，靠 **模仿學習（imitation learning）+ RL** 學會走路，
最後把訓練出來的策略部署到 Raspberry Pi Zero 2W 上實機執行。

---

## 四個子專案

| 目錄 | 角色 | 預設分支 | 技術棧 |
|---|---|---|---|
| `Open_Duck_Mini/` | **Hub / 硬體本體**：CAD、BOM、文件、機構設計、示範用的 onnx policy | `v2` | Python, MuJoCo, Onshape |
| `Open_Duck_reference_motion_generator/` | **產動作**：用 Placo 解算步態，產生模仿學習用的 reference motion | `main` | Python, [Placo](https://github.com/Rhoban/placo), git-lfs |
| `Open_Duck_Playground/` | **訓練**：MuJoCo Playground + Brax PPO，吃 reference motion 當 imitation reward | `main` | JAX, Brax, MuJoCo MJX, uv |
| `Open_Duck_Mini_Runtime/` | **上機執行**：在 Raspberry Pi Zero 2W 上跑 onnx policy、驅動馬達/IMU/手把 | `v2` | Python, onnxruntime, rustypot |

> `Open_Duck_Mini` 與 `Open_Duck_Mini_Runtime` 的預設分支是 **`v2`**，不是 `main`。

---

## 資料流（這是理解整套系統的主線）

```
┌─────────────────────────────────────────┐
│ Open_Duck_reference_motion_generator    │  ① 產動作
│   scripts/auto_waddle.py  --sweep       │
│     → recordings/*.json                 │
│   scripts/fit_poly.py --ref_motion ...  │
│     → polynomial_coefficients.pkl       │
└───────────────────┬─────────────────────┘
                    │  複製 polynomial_coefficients.pkl
                    ▼
┌─────────────────────────────────────────┐
│ Open_Duck_Playground                    │  ② 訓練
│   playground/open_duck_mini_v2/data/    │  ← 放 .pkl
│   joystick.py: USE_IMITATION_REWARD=True│
│   runner.py  → PPO 訓練                  │
│   common/export_onnx.py → policy.onnx   │
│   mujoco_infer.py -o policy.onnx        │  （先在模擬裡驗證）
└───────────────────┬─────────────────────┘
                    │  複製 policy.onnx（＋同一份 .pkl）
                    ▼
┌─────────────────────────────────────────┐
│ Open_Duck_Mini_Runtime  (on Rasp Zero 2W)│ ③ 上機
│   scripts/v2_rl_walk_mujoco.py           │
│     --onnx_model_path <policy.onnx>      │
│   OnnxInfer → 50Hz 控制迴圈 → rustypot   │
│   ↑ 讀 ./polynomial_coefficients.pkl     │
└──────────────────────────────────────────┘
                    ▲
                    │  機構 / BOM / CAD / MJCF 來源
┌───────────────────┴─────────────────────┐
│ Open_Duck_Mini（hub，非流程節點）         │
└──────────────────────────────────────────┘
```

### 關鍵：兩個跨 repo 的交接檔案

1. **`polynomial_coefficients.pkl`** — 步態的多項式擬合係數。
   由 generator 產生，**同一份**必須同時出現在：
   - `Open_Duck_Playground/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl`（訓練時算 imitation reward）
   - `Open_Duck_Mini_Runtime/mini_bdx_runtime/mini_bdx_runtime/polynomial_coefficients.pkl` 以及執行目錄下的 `./polynomial_coefficients.pkl`（runtime 用它組 observation）

   ⚠️ **訓練與 runtime 用的 .pkl 必須是同一份。** 不一致 → observation 對不上 → 實機亂走。

2. **`*.onnx`** — 訓練完的 policy。Playground 匯出 → 複製到樹莓派 → runtime 載入。
   `Open_Duck_Mini/BEST_WALK_ONNX.onnx`、`BEST_WALK_ONNX_2.onnx` 是官方已訓練好、可直接拿來跑的版本。

---

## 常用指令速查

### ① 產 reference motion
```bash
cd Open_Duck_reference_motion_generator
git lfs pull                                  # STL 沒下載會報 mesh directory 錯誤
uv run scripts/auto_waddle.py -j8 --duck open_duck_mini_v2 --sweep
uv run scripts/fit_poly.py --ref_motion recordings/
uv run scripts/plot_poly_fit.py --coefficients polynomial_coefficients.pkl   # 檢查擬合品質
uv run scripts/replay_motion.py -f recordings/<file>.json                    # 視覺化單一動作
```

### ② 訓練
```bash
cd Open_Duck_Playground
cp ../Open_Duck_reference_motion_generator/polynomial_coefficients.pkl \
   playground/open_duck_mini_v2/data/
# 確認 playground/open_duck_mini_v2/joystick.py 內 USE_IMITATION_REWARD=True
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
uv run tensorboard --logdir=<yourlogdir>
uv run playground/open_duck_mini_v2/mujoco_infer.py -o <path_to.onnx>       # 模擬驗證
```

### ③ 上機
```bash
# 在 Raspberry Pi Zero 2W 上（見 Open_Duck_Mini_Runtime/README.md 的完整燒錄/接線步驟）
workon open-duck-mini-runtime
cd Open_Duck_Mini_Runtime/scripts
python v2_rl_walk_mujoco.py --onnx_model_path ~/BEST_WALK_ONNX_2.onnx -c 50
```
輔助腳本：`check_motors.py`、`check_voltage.py`、`calibrate_imu.py`、`find_soft_offsets.py`、
`configure_all_motors.py`、`turn_on.py` / `turn_off.py`、`head_puppet.py`。

---

## 工作時的注意事項

- **不要跨 repo 直接改檔案來「同步」**：四個 repo 各自 push 到不同 remote。改動請在對應 repo 裡 commit。
- **generator 用 git-lfs**：clone 後若機器人 mesh 讀不到，先 `git lfs pull`。
- **Playground / generator 用 `uv`**，Runtime 用 `virtualenvwrapper`（樹莓派上）——不要混用。
- **加新機器人**到 Playground：複製 `playground/open_duck_mini_v2/` 後改 `base.py`、`constants.py`、
  `joystick.py`、`runner.py`，並把 MJCF 放進 `xmls/`。
- **`Open_Duck_Mini_Runtime/README_LOREN.md`** 是中文的逐檔案說明（內容其實在講 Playground 的結構），
  找 `playground/common/*` 各檔案用途時可以參考。
- Runtime 控制頻率預設 **50Hz**、`action_scale` 預設 **0.25**，這兩個值必須和訓練時一致。

## 上游來源

四個 repo 都 fork 自 [apirrone](https://github.com/apirrone) 的 Open Duck 專案系列，
目前 remote 指向 `lorenhsu1128/*`。要跟上游同步時記得手動加 upstream remote。
