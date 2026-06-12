# V-JEPA2 輸送帶循環任務設計

## Isaac Sim 資產

Isaac Sim 5.1 提供 Conveyor Track Builder，可使用內建的直線、彎道、起點、
終點與滾輪式輸送帶資產。官方資產通常位於：

```text
[Isaac Assets Root]/Isaac/Props/Conveyors/
```

目前腳本使用程式產生的半圓形輸送帶，並套用 PhysX 表面速度。輸送帶區段
設為 kinematic rigid body，因此皮帶本體不會因重力散開，但仍能推動方塊。

障礙物使用 Isaac Sim 的動態幾何物件建立。後續若改用 RMPflow，可將立方體、
球體、膠囊體與地面註冊為障礙物，並透過 `update_world()` 更新位置。

## 循環流程

1. xArm6 位於半徑 `0.43 m`、涵蓋約 `210` 度的弧形輸送帶中央。
2. 方塊先放在輸送帶終點，輸送帶保持停止，確認方塊不會漂移。
3. 手臂下降夾住方塊，先垂直抬高，再由障礙物外側繞行。
4. 手臂將方塊放到輸送帶起點，放開夾爪並返回初始姿勢。
5. 輸送帶以預設 `0.25 m/s` 啟動，將方塊送回終點。
6. 只有方塊回到終點範圍，才算完成一個循環；之後輸送帶停止。

每個 episode 會改變障礙物的位置與尺寸。教師控制器目前使用 IK 與明確的
抬升、繞行路徑產生避障示範；這不代表尚未訓練的模型已具備自主避障能力。

物理驗證會檢查：

- 初始夾爪碰撞幾何最低點必須高於地面。
- 全部機器人剛體 link 與障礙物必須保持至少 `12 mm` 的安全距離。
- 基座、大臂與前臂會對遠端腕部及夾爪檢查至少 `20 mm` 的自碰撞距離，
  避免手臂向內折疊得太近；緊湊腕部內原本相鄰的 link 不納入此門檻。
- 方塊在終點等待時的漂移量。
- 方塊是否確實被夾起。
- 方塊是否被放到輸送帶起點。
- 方塊是否由輸送帶送回終點。
- 最大夾取高度、回程時間與最終距離。

非 headless 模式會顯示 `xArm6 Test Control` 視窗。循環完成並顯示
`Ready - click the button to run again` 後，按 `Run Test Again`，會使用
相同的隨機種子與障礙物配置重新執行一次。

## 預覽指令

可從任何 PowerShell 目錄執行：

```powershell
& C:\Users\User\isaac_sim_5.1\python.bat `
  C:\Users\User\Documents\XArm\scripts\conveyor_cycle_scene.py `
  --preview-only `
  --conveyor-speed 0.25
```

若要停用輸送帶圖形設定：

```powershell
& C:\Users\User\isaac_sim_5.1\python.bat `
  C:\Users\User\Documents\XArm\scripts\conveyor_cycle_scene.py `
  --preview-only `
  --no-conveyor-graph
```

## 產生訓練資料

以下範例產生 `100` 個 episode：

```powershell
& C:\Users\User\isaac_sim_5.1\python.bat `
  C:\Users\User\Documents\XArm\scripts\conveyor_cycle_scene.py `
  --headless `
  --episodes 100 `
  --seed 1000 `
  --conveyor-speed 0.25 `
  --record-root C:\Users\User\Documents\XArm\outputs\conveyor_dataset
```

每個 episode 可包含：

- `rgb_*.png`：相機 RGB 畫面。
- `actions.jsonl`：關節、夾爪、物體與任務階段資料。
- `metadata.json`：場景、障礙物、驗證結果及執行參數。

## V-JEPA2 訓練入口

先驗證資料集：

```powershell
& C:\Users\User\isaac_sim_5.1\python.bat `
  C:\Users\User\Documents\XArm\scripts\train_vjepa2_ac.py `
  --dataset-root C:\Users\User\Documents\XArm\outputs\conveyor_dataset `
  --dry-run
```

再啟動 encoder 與 predictor 訓練：

```powershell
& C:\Users\User\isaac_sim_5.1\python.bat `
  C:\Users\User\Documents\XArm\scripts\train_vjepa2_ac.py `
  --dataset-root C:\Users\User\Documents\XArm\outputs\conveyor_dataset `
  --model facebook/vjepa2-vitl-fpc64-256 `
  --batch-size 1 `
  --epochs 20
```
