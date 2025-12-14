# VSR関連論文まとめ

## 1. UltraVSR (2505.19958)

**タイトル**: UltraVSR: Achieving Ultra-Realistic Video Super-Resolution with Efficient One-Step Diffusion Space

**著者**: Yong Liu et al. (Xi'an Jiaotong University, Huawei Noah's Ark Lab)

**発表**: MM '25 (ACM Multimedia 2025)

### 概要
拡散モデルを用いた効率的な1ステップVideo Super-Resolution (VSR)フレームワーク。従来の拡散ベースVSR手法が抱える「時間的一貫性の欠如」と「計算コストの高さ」を解決。

### 課題
- 拡散モデルの確率的な性質により、フレーム間でフリッカー（ちらつき）が発生
- 従来手法は複数のサンプリングステップが必要で推論が遅い
- 低解像度動画からの動き推定が信頼性に欠ける

### 提案手法

#### 1. Degradation-aware Reconstruction Scheduling (DRS)
- 入力LR動画から劣化度(degradation factor) `d`を推定
- 劣化度と拡散ノイズスケジュールを直接マッピング
- **1ステップでLRからHRへの再構成を実現**
- CLIPモデルを使用して劣化度を推定

#### 2. Recurrent Temporal Shift (RTS) モジュール
時間的整合性を確保するための軽量モジュール：
- **RTS-Convolution Unit**: 特徴マップをチャネル方向に3分割し、時間方向にシフトして隣接フレームから情報を取得
- **RTS-Attention Unit**: Key/Valueテンソルを時間方向にシフトし、Mutual Self-Attentionで意味的情報を統合

#### 3. Spatio-temporal Joint Distillation (SJD)
2つのTemporal Regularized UNet (TRU)を使用：
- **Real TRU**: Ground Truth動画の分布を学習
- **Fake TRU**: 合成出力動画の分布を学習
- Content-realistic gradient lossとTemporal-consistency gradient lossで最適化

#### 4. Temporally Asynchronous Inference (TAI)
- 長い動画を複数のサブシーケンスに分割
- 空間的処理と時間的処理を分離
- メモリ効率を大幅に改善（2K解像度で120フレームまで処理可能）

### 実験結果
- **速度**: 既存拡散ベースVSRの0.06〜0.08倍の推論時間（720p: 0.89秒、2K: 2.67秒）
- **品質**: VideoLQベンチマークで全ての非参照品質指標（MUSIQ, CLIP-IQA, MANIQA, DOVER）でSOTA
- **パラメータ**: 訓練可能パラメータはわずか10.5M

### キーポイント
- 1ステップ拡散でVSRを実現した初の研究
- 明示的な時間層なしで時間的整合性を達成
- リアルタイムに近い推論速度と高品質を両立

---

## 2. DiTVR (2508.07811)

**タイトル**: DiTVR: Zero-Shot Diffusion Transformer for Video Restoration

**著者**: Sicheng Gao, Nancy Mehta, Zongwei Wu, Radu Timofte (University of Würzburg)

**発表**: AAAI 2026

### 概要
事前学習済みDiffusion Transformer (DiT)を活用したゼロショットビデオ復元フレームワーク。オプティカルフローに基づく軌跡追跡と特殊なアテンション機構により、タスク固有の学習なしで高品質なビデオ復元を実現。

### 課題
- 回帰ベースの手法は非現実的なディテールを生成し、大規模ペアデータセットが必要
- 画像拡散モデルを動画に適用すると各フレームが独立したノイズパスを辿り、フリッカーが発生
- U-Netベースの手法は長距離依存性の処理が困難
- 単純なオプティカルフローワーピングはトークンのミスアライメントを引き起こす

### 提案手法

#### 1. Spatiotemporal Neighbor Selection Cache (STNC)
- 非重複空間ブロックからKey-Value表現を保存・取得
- オプティカルフロー軌跡に基づいて動的にトークンを選択
- 空間的隣接と時間的隣接の両方を考慮
- メモリ使用量を削減しながら長距離時間的モデリングを実現

#### 2. Trajectory-Aware Attention (TAttn)
- フレーム単位のアテンションの代わりに、動き軌跡に沿って特徴を整列
- **Vital Layer分析**: 時間的ダイナミクスに最も敏感なレイヤーを特定
- 2段階アテンション：
  1. ブロック内Self-Attention
  2. 空間的/時間的隣接間のCross-Attention
- ゴースティングアーティファクトを軽減

#### 3. Flow-Guided Diffusion Sampler
ウェーブレット領域でのフロー対応補正：
- **Step 1 (Wavelet Split)**: DWTで低周波・高周波サブバンドに分解
- **Step 2 (Low-Frequency Data Fidelity)**: DDNMのrange/null-space哲学に基づき、低周波部分にのみ劣化演算子を適用
- **Step 3 (Flow-guided Residual Alignment)**: フロー軌跡に沿って残差を平均化し、フレーム間ドリフトを抑制

### Optical Flow Trajectory Sampling
- GMFlowモデルで双方向オプティカルフロー推定
- 双方向検証により安定した非オクルージョン動き経路のみを使用
- Transformerベースのため1回のみダウンサンプリングし、情報損失を防止

### 実験結果
- **評価タスク**: Video SR (4×)、Blind Video SR、Video Denoising
- **データセット**: DAVIS, SPMCS, Vid4
- **指標**: PSNR, SSIM, LPIPS, Warping Error (WE), Frame Similarity (FSim)
- ゼロショット手法としてDAVISで最高性能（PSNR: 33.29, WE: 10.20×10⁻³）
- DiffIR2VR-Zeroを上回る時間的一貫性

### キーポイント
- DiT（Diffusion Transformer）をビデオ復元に初適用したゼロショットフレームワーク
- U-Netの限られた受容野問題を解決
- 学習不要で複数の復元タスク（SR、デノイジング、デブラー）に対応
- Vital Layer分析により効率的に時間的整合性を維持

---

## 3. VDM-VSR (2503.03355)

**タイトル**: Rethinking Video Super-Resolution: Towards Diffusion-Based Methods without Motion Alignment

**著者**: Zhihao Zhan et al. (TopXGun Robotics, Nanjing University)

**発表**: arXiv 2025

### 概要
Diffusion Posterior Sampling (DPS) フレームワークと無条件ビデオ拡散トランスフォーマーを組み合わせた新しいVSRアプローチ。**明示的なモーション推定やピクセルアライメントを必要としない**ことが最大の特徴。

### 核心的な主張
強力なビデオ拡散モデル（VDM）が実世界の物理を学習すれば、様々な動きパターンを事前知識として自然に処理できる。顔画像生成で顔の対称性が自然に維持されるように、VDMは明示的な動き推定なしでフレーム間の整合性を保てる。

### 従来手法の課題
1. **動き推定ありの手法**: オプティカルフロー推定のエラーがアーティファクトを生成、複雑な動き（回転、オクルージョン、非剛体）に弱い
2. **動き推定なしの手法**: 長期的なピクセル依存性を捉えられず、復元効率が低い
3. **アライシングのジレンマ**: 高周波成分保持にはエイリアシングが必要だが、空間整列にはエイリアシング低減が必要

### 提案手法: VDM-VSR

#### アーキテクチャ
- **DPSフレームワーク**: 逆問題として定式化（MAP推定に類似）
- **VAE潜在空間**: 計算資源節約のため空間方向に8倍圧縮
- **DiTベースデノイザー**: OpenSoraプロジェクトのSTDiTに類似（条件付け埋め込みなし）

#### 劣化モデル
```
Y = H(X, h) + E
```
- フレームごとにブラーカーネル畳み込み + ダウンサンプリング
- フレームマスキング機能も追加可能

#### 逆拡散プロセス
1. 潜在空間でガウスノイズからスタート
2. 無条件VDMでデノイズ
3. VAEデコード後、観測LRとの差分を勾配として補正
4. 繰り返して最終的にHR動画を生成

### 実験結果

#### Moving MNIST（合成データ）
- **フレーム数と品質の関係**: フレーム数増加でPSNRが向上（1→5フレームで着実に改善）
- **ランダム順 vs 順次順**: ランダム順の方がPSNR向上が速い（時間軸で離れたフレームが動き把握に有効）
- **エイリアシングの影響**: ブラーカーネルσhに関わらず、十分なフレーム数で同等の品質に収束

#### BAIRデータセット（実世界ロボット）
| Method | PSNR | SSIM |
|--------|------|------|
| IART | 24.80 | 0.886 |
| FMA-Net | 25.35 | 0.897 |
| **VDM-VSR** | **27.44** | **0.928** |

### キーポイント
- 明示的なモーション推定が不要な初のDPS+VDMベースVSR
- 3D動画全体を空間・時間軸で統一的に処理
- フレーム数増加でエイリアシング問題を緩和可能
- 現状は計算リソースの制限により大規模実用化には至らず

### 限界
- 大規模VDM（Sora級）が必要で、現状の計算リソースでは汎用解決策にならない
- 学習データと計算資源の制約

---

## 4. FCVSR (2502.06431)

**タイトル**: FCVSR: A Frequency-aware Method for Compressed Video Super-Resolution

**著者**: Qiang Zhu, Fan Zhang, Feiyu Chen, Shuyuan Zhu, David Bull, Bing Zeng (UESTC, University of Bristol)

**発表**: arXiv 2025

### 概要
周波数領域での時空間情報を活用した圧縮ビデオ超解像モデル。既存の周波数ベース手法が「周波数サブバンドの空間的な差別化」や「時間的な周波数ダイナミクスの捕捉」を行っていない問題を解決。

### 課題
- 既存の周波数ベースVSR手法は異なる周波数サブバンドを空間的に区別しない
- 時間的な周波数ダイナミクスを捉えられていない
- 圧縮アーティファクトとダウンサンプリング劣化の両方への対応が困難

### 提案手法

#### 1. Motion-Guided Adaptive Alignment (MGAA) モジュール
周波数領域でフレーム間の動きを推定し、適応的畳み込みで特徴アライメント：

**Motion Estimator**:
- FFTで特徴を周波数領域に変換
- 実部と虚部を連結
- 異なるカーネルサイズの複数ブランチで多様な動きオフセット `{on}` を学習
- 相関演算（Correlation Operation）で特徴間の対応関係を取得

**Kernel Predictor**:
- 適応的畳み込みカーネル `K = {Kn}` を予測
- 各カーネルは2つの1次元カーネル（垂直・水平）で構成

**Motion-Guided Adaptive Convolution (MGAC) Layer**:
- 推定した動きオフセットに基づいて空間サンプリング
- 分離可能フィルタで大きな受容野を低コストで実現
- N個の適応的畳み込みをカスケード接続

#### 2. Multi-Frequency Feature Refinement (MFFR) モジュール
分解-強調-集約戦略で高周波詳細を復元：

**Decoupler**:
- ガウシアンバンドパスフィルタで入力特徴をQ個の周波数サブバンドに分解
- FFTで周波数領域に変換後、マスクを適用

**Enhancer**:
- **FFE (Feedforward Enhancement) ブランチ**: 低周波サブバンドの和からSqを減算して高周波特徴を取得
- **FBE (Feedback Enhancement) ブランチ**: 強調済み特徴を使って高周波特徴を生成
- 低周波から高周波へ段階的に強調

**Aggregator**:
- 全ての強調済み特徴を加算
- チャネルアテンション（CA）で統合

#### 3. Frequency-aware Contrastive (FC) Loss
2D離散ウェーブレット変換（DWT）と対照学習に基づく損失関数：
- **Positive sets**: GTフレームの高周波サブバンド（HH, HL, LH）と低周波サブバンド（LL）
- **Negative set**: アップサンプルされた圧縮フレームの高周波サブバンド
- **Anchor sets**: SR結果の各サブバンド
- 高周波成分の復元を促進

### アーキテクチャ
```
入力: 7フレーム {Ii}(t-3〜t+3)
↓ Conv
↓ MGAA × 3 (階層的アライメント)
↓ MFFR (周波数詳細強化)
↓ REC (再構成)
↓ + Bilinear Upsampled
出力: HR frame
```

### 実験結果

#### REDS4 (QP=37) での比較
| Method | PSNR | SSIM | VMAF | Params | FLOPs |
|--------|------|------|------|--------|-------|
| BasicVSR++ | 25.05 | 0.6620 | 31.25 | 7.32M | 395.69G |
| FTVSR++ | 25.09 | 0.6625 | 31.81 | 10.80M | 1148.85G |
| IA-RT | 25.16 | 0.6678 | 32.01 | 13.41M | 1939.50G |
| **FCVSR-S** | 24.93 | 0.6611 | 31.56 | **3.70M** | **68.82G** |
| **FCVSR** | **25.20** | **0.6694** | **32.05** | 8.81M | 165.36G |

#### CRFベース圧縮での比較（REDS4 CRF=25）
| Method | PSNR | SSIM |
|--------|------|------|
| FTVSR++ | 28.28 | 0.783 |
| **FCVSR** | **28.36** | **0.788** |

### アブレーション結果
- MGAAモジュール: FGDAと比較して+0.10dB PSNR、-1.64M params、-45.28G FLOPs
- MFFRモジュール: 分解数Q=8が性能と複雑さのバランスが最適
- FC Loss: 高周波・低周波両方のサブロスが性能向上に寄与

### キーポイント
- 周波数領域での動き推定とアライメントを初めてVSRに適用
- 複数の動きオフセットで多様な動きパターンに対応
- ガウシアンバンドパスフィルタによる明示的な周波数サブバンド分解
- 対照学習で高周波詳細の復元を促進
- 高性能かつ低複雑度（FCVSR-Sは3.70M params、68.82G FLOPs）

---

## 5. RCDM (2502.01816)

**タイトル**: Low-Resource Video Super-Resolution using Memory, Wavelets, and Deformable Convolutions

**著者**: Kavitha Viswanathan, Amit Sethi et al. (IIT Bombay)

**発表**: arXiv 2025

### 概要
わずか2.3Mパラメータで SOTA性能を達成する軽量VSRアーキテクチャ。Transformerベース手法の7〜35倍少ない計算量で、エッジデバイスでのリアルタイムVSRを目指す。

### 課題
- Transformerベースモデル（VRT, RVRT）は高性能だが計算コストが高い
- 既存の軽量モデルは時間的一貫性や大きな動きへの対応が困難
- 多くの手法が空間・時間領域のみに焦点を当て、周波数領域表現を活用していない

### 提案手法: RCDM (Residual ConvNeXt Deformable Convolution with Memory)

#### 問題定式化
```
入力: 2N+1 LRフレーム {It-N, ..., It, ..., It+N}
出力: HR中央フレーム Ht, 次フレーム用メモリ Mt
Ht, Mt = fα(It-N, ..., It, ..., It+N, Mt-1)
```

#### 1. Alignment Block (Deformable Convolution)
- 明示的なオプティカルフロー推定なしで動き補償
- 3D Deformable Convolutionで空間・時間両方の動きをモデリング
- 学習可能なオフセットでサンプリング位置を動的に調整

#### 2. Residual Spatio-Temporal Feature Extraction
- Residual 3D Convolution Blockで複数フレーム情報を集約
- 参照フレームの構造を保持しながら時空間特徴を抽出

#### 3. Wavelet-Based Multi-Scale Feature Extraction
- 2D離散ウェーブレット変換（DWT）で中央フレームを周波数サブバンドに分解
- 水平・垂直・対角線の詳細を独立に処理
- 特徴サイズ: (h, w, c) → (h/2, w/2, 4c)（情報損失なし）
- テクスチャ復元と高周波詳細の保持を強化

#### 4. Memory Mechanism for Temporal Consistency
メモリテンソルで長期的な時間一貫性を維持：
```
Mt = β * Mt-1 + Ht^feat
```
- β: 過去フレームの影響を制御する学習可能な重み
- Ht^feat: 現フレームの精製済み特徴
- フリッカーアーティファクトを防止

#### 5. Super-Resolution and Upsampling
- **ConvNeXt Blocks**: Depthwise convolution + 残差接続で効率的な特徴精製
- **Pixel Shuffle**: 特徴マップを再配置してHRフレームを再構成

### アーキテクチャ構成
```
LR images (5フレーム)
    ↓
Alignment Block (3D Deformable Conv)
    ↓
┌─────────────────────────────────────┐
│ Residual Blocks (3D Conv)           │
│ Wavelet Block (2D DWT → Upsample)   │ ← 並列パイプライン
└─────────────────────────────────────┘
    ↓ 融合
Super Resolution Block
    ↓
ConvNeXt + Pixel Shuffling
    ↓
Memory (t-1) → 統合 → HR (t)
```

### モデルバリエーション
| Model | Params | GFLOPs | SSIM (REDS4) |
|-------|--------|--------|--------------|
| RCDM-light | 1.86M | 203.69 | 0.9041 |
| RC2DMdwtState | 2.10M | 262.05 | 0.9075 |
| RC2DM | 2.13M | 272.57 | 0.9066 |
| **RCDM** | **2.30M** | **281.06** | **0.9175** |
| RCDMdwtState | 2.56M | 353.27 | 0.9129 |

### 実験結果

#### 他手法との比較（REDS4 BI Degradation）
| Method | Params | GFLOPs | SSIM |
|--------|--------|--------|------|
| BasicVSR++ | 7.32M | 1962 | 0.9069 |
| RVRT | 11.06M | 2682 | 0.9113 |
| EvTexture | 8.90M | 4148 | 0.9174 |
| VRT | 28.80M | 2682 | 0.9006 |
| **RCDM (Ours)** | **2.30M** | **281** | **0.9175** |

- BasicVSR++より+0.0106 SSIM向上、パラメータ1/3、FLOPs 1/7
- RVRTより+0.0062 SSIM向上、パラメータ1/5、FLOPs 1/10

### アブレーション結果
- **Memory Mechanism**: 後半フレームほど効果が大きい（証拠蓄積効果）
- **Wavelet-Based Feature Extraction**: 高周波詳細保持とテクスチャ復元に貢献
- 両コンポーネントは相補的: メモリは時間遷移を滑らかに、ウェーブレットは構造的整合性を向上

### キーポイント
- VSRにはグローバルSelf-Attentionは不要、ローカル特徴抽出が効果的
- メモリテンソルで時間的一貫性を確保（ConvLSTMより計算効率的）
- 2Dウェーブレット分解でエッジのスパース性を活用
- 複数の並列ブランチで時空間情報の様々な側面を活用
- エッジデバイスでのリアルタイムVSRに最適

### 限界と今後の方向性
- 詳細保持と構造再構成のさらなる改善が必要
- Swin Transformerのようなローカル限定Self-Attentionの導入可能性

---

## 6. RepNet-VSR (2504.15649)

**タイトル**: RepNet-VSR: Reparameterizable Architecture for High-Fidelity Video Super-Resolution

**著者**: Biao Wu, Diankai Zhang, Shaoli Liu, Si Gao, Chengjian Zheng, Ning Wang (ZTE)

**発表**: arXiv 2025 (MAI2025 Challenge)

### 概要
モバイルNPUでのリアルタイム4倍VSRを目指した再パラメータ化アーキテクチャ。REDS検証セットで27.79 dB PSNRを達成し、MediaTek Dimensity NPUで10フレームあたり103msで180p→720pの処理が可能。

### 課題
- 深層CNNは高性能だがモバイルデバイスでの実行が困難
- VSRの計算集約的な性質がエッジデバイスへの展開を制限
- 既存手法は再構成品質か計算効率のどちらかに偏っている

### チャレンジの制約条件
```
入力: [1×180×320×30] (10フレーム × 3チャネル)
出力: [1×720×1280×30] (4倍アップスケーリング)

スコア計算式:
Score(PSNR, runtime) = 2^(2*(PSNR-27)) / runtime
```
→ 計算効率を2倍にすると、PSNR 0.5dB向上と同等の効果

### 提案手法

#### 1. ネットワークアーキテクチャ
- **Multi-level Feature Fusion**: チャネル連結で深層・浅層特徴を統合
- **1×1畳み込み**: 次元削減で計算効率を最適化
- **Depth-to-Space前の最適化**: 3×3畳み込みを1×1畳み込みに置換（48→3チャネル圧縮）
  → アップサンプリングが4倍高速化

#### 2. RepConv（再パラメータ化モジュール）
**学習時（複雑な構造）**:
```
入力
  ├─ 1×1 Conv (チャネル4倍拡張)
  │     ↓
  │   3×3 Conv (空間特徴抽出)
  │     ↓
  │   1×1 Conv (チャネル削減)
  │     ↓
  └─────⊕ (残差接続: 1×1 Conv)
        ↓
      ReLU
```

**推論時（単一畳み込みにマージ）**:
```
入力 → Conv 3×3 → ReLU → 出力
```

#### 3. Neural Architecture Search (FGNAS)
PSNRとFLOPsの最適バランスを自動探索：
- **探索空間**: チャネル数 {0-32}、RepBlock数 {0-8}
- **目的関数**: `min L(ψ,θ) + λ·R(ψ)` （タスク損失 + 計算コストペナルティ）
- **最適構成**: 16チャネル、4 RepBlock

### アーキテクチャ全体
```
LR frames (10フレーム)
    ↓
RepConv → RepConv → ... → RepConv
    ↓
[特徴連結] ← 深層・浅層特徴
    ↓
Conv 1×1 (次元削減)
    ↓
Conv 1×1 (48→3チャネル)
    ↓
Depth2Space (4倍アップスケーリング)
    ↓
HR frames
```

### 学習設定
- **2段階学習**:
  1. NASによるアーキテクチャ探索
  2. L2損失によるファインチューニング
- オプティマイザ: Adam (β1=0.9, β2=0.999)
- 学習率: 5×10⁻⁴（500エポックウォームアップ後、線形減衰）
- パッチサイズ: HR 384×384、LR 96×96

### 実験結果

#### MediaTek NPUでの比較
| Model | Year | PSNR | CPU Runtime | NPU Runtime | Score |
|-------|------|------|-------------|-------------|-------|
| EVSRNet | MAI2021 | 27.42 | 271ms | 103ms | 0.0174 |
| RCBSR | MAI2022 | 27.28 | 112ms | 95.8ms | 0.0154 |
| **RepNet-VSR** | **MAI2025** | **27.79** | 273ms | **103ms** | **0.029** |

- EVSRNetより+1.8% PSNR向上、スコア66.7%向上
- RCBSRより+1.9% PSNR向上、スコア88.3%向上

#### 異なる構成の比較
| nc | nb | PSNR | Runtime(ms) | Score |
|----|-------|------|-------------|-------|
| 8 | 4 | 27.39 | 56.5 | 0.0303 |
| **16** | **4** | **27.79** | **89.6** | **0.0334** |
| 32 | 4 | 28.01 | 149.7 | 0.0271 |
| 16 | 5 | 27.83 | 93.6 | 0.0323 |

### キーポイント
- 再パラメータ化で学習時の複雑なネットワークを推論時に単純化
- NASでPSNRと計算効率の最適バランスを自動探索
- 1×1畳み込みによる効率的なチャネル圧縮
- MAI Video Super-Resolution Challengeの歴代優勝アルゴリズムを上回るスコア
- モバイルNPUでのリアルタイム推論を実現

---

## 7. SDATC (2502.07381)

**タイトル**: Spatial Degradation-Aware and Temporal Consistent Diffusion Model for Compressed Video Super-Resolution

**著者**: Hongyu An, Xinfeng Zhang et al. (University of Chinese Academy of Sciences, ByteDance, Peking University)

**発表**: arXiv 2025

### 概要
圧縮ビデオ超解像に特化した拡散モデル。事前学習済みLDMs（Latent Diffusion Models）の生成能力を活用し、圧縮アーティファクトを考慮した空間的劣化認識と時間的一貫性を両立。

### 課題
- 一般的なVSR手法は圧縮ビデオを想定していない
- 圧縮アーティファクトを本物のテクスチャと誤認識して増幅してしまう
- 量子化による高周波情報の損失は復元困難
- 拡散モデルをそのまま適用すると時間的一貫性が損なわれる

### 提案手法: SDATC

#### 1. Distortion Control Module (DCM)
- LQ入力フレームを前処理し、拡散モデルへの入力を調整
- Residual Swin Transformer Blocks (RSTB) + PixelShuffle で歪み除去・アップサンプリング
- ControlNetに類似したzero convolutionでガイダンスを生成
- 圧縮ノイズが生成プロセスに悪影響を与えることを防止

#### 2. Compression-Aware Prompt Module (CAPM)
- 圧縮情報をプロンプトとしてエンコード
- 補助エンコーダで潜在特徴を圧縮空間に変換
- UNetデコーダとVAEデコーダの各レベルに挿入
- 異なる圧縮レベル（CRF値）に適応的に対応
- ランダム初期化プロンプトより精度の高い劣化推定

#### 3. Spatio-Temporal Attention Module (STAM)
VAEデコーダに時間次元を拡張：
- **3D CNN**: 時空間特徴を抽出
- **Temporal Attention Block (TAB)**: 時間方向のself-attention
- 学習可能パラメータ αT, βT で空間・時間ブランチのバランス調整
- Controllable Feature Warping (CFW) でfidelityを制御

#### 4. Optical Flow-Based Alignment
- 各デノイズステップでRAFTを用いて前後方向オプティカルフローを計算
- Motion error Et を計算し、サンプリングプロセスに勾配として反映
- フレーム間ドリフトを抑制

#### 5. Color Correction
- AdaIN（Adaptive Instance Normalization）で色シフト問題を解決
- LQフレームの平均・分散を参照して正規化

### 実験設定
- **学習**: REDSデータセット（266シーケンス）
- **圧縮**: x264エンコーダ、CRF = 15, 25, 35
- **評価データセット**: REDS4, Vid4, UDM10

### 実験結果
- 知覚品質指標（LPIPS, DISTS, FID, NIQE, MANIQA, CLIP-IQA）で包括的に優位
- 特にMANIQAとCLIP-IQAでほぼ全てのケースで最高スコア
- PSNR/SSIMでは生成モデルの特性上やや劣るが、他の生成手法より優れる

### REDS4 CRF=25での比較例
| Method | LPIPS↓ | DISTS↓ | NIQE↓ | MANIQA↑ |
|--------|--------|--------|-------|---------|
| BasicVSR++ | 0.4822 | 0.1861 | 6.019 | 0.1681 |
| MGLD-VSR | 0.3366 | 0.0975 | 2.964 | 0.3703 |
| **SDATC** | **0.3488** | **0.0894** | **2.796** | **0.3796** |

### キーポイント
- 圧縮ビデオVSRに特化した初の包括的拡散フレームワーク
- DCMで入力の歪みを制御し、生成品質を向上
- CAPMで圧縮レベルに適応的なプロンプト学習
- STAMとオプティカルフローで時間的一貫性を確保
- Stable Diffusion v2.1をバックボーンとして使用

---

## 8. TDM (2501.02269)

**タイトル**: TDM: Temporally-Consistent Diffusion Model for All-in-One Real-World Video Restoration

**著者**: Yizhou Li, Zihua Liu, Yusuke Monno, Masatoshi Okutomi (Institute of Science Tokyo)

**発表**: arXiv 2025

### 概要
初の拡散モデルベースAll-in-Oneビデオ復元手法。事前学習済みStable DiffusionとファインチューニングしたControlNetを活用し、複数の劣化タイプを単一モデルで処理。

### 課題
- 従来手法は劣化タイプごとに専用モデルが必要
- 拡散モデルの確率的性質がフレーム間の時間的一貫性を損なう
- 複数フレームを同時処理する学習は膨大なメモリを要する
- 実世界データへの汎化性能が不十分

### 提案手法: TDM

#### 1. Task Prompt Guidance (TPG)
タスク固有のテキストプロンプトで拡散プロセスを誘導：
- **Dehazing**: "remove the fog"
- **Deraining**: "remove the rain"
- **Denoising**: "remove the noise"
- **Super-Resolution**: "recover the high resolution details"
- **MP4圧縮アーティファクト除去**: タスク固有プロンプト

事前学習済みStable Diffusionのゼロショット分類能力を活用し、追加パラメータや計算なしで複数タスクに対応。

#### 2. 単一画像入力での学習
- ControlNetを単一画像入力でファインチューニング
- 時間的一貫性は推論時のtraining-free操作で確保
- 単一GPU（RTX 4090）で学習可能
- 既存の単一画像復元データセットをそのまま利用可能

#### 3. Sliding Window Cross-Frame Attention (SW-CFA)
従来のcross-frame attention（第1フレームを参照）の問題を解決：

```
従来手法: Q = W^Q z^{v_i}, K = W^K z^{v_0}, V = W^V z^{v_0}
↓
SW-CFA: K = (1/(2N+1)) Σ_{j=i-N}^{i+N} K_j, V = (1/(2N+1)) Σ_{j=i-N}^{i+N} V_j
```

- スライディングウィンドウ内のKey/Valueを平均化
- 時間的なローパスフィルタとして機能
- 大きな動きにも対応可能
- 計算オーバーヘッドは最小限

#### 4. DDIM Inversion
- ランダムガウシアンノイズではなく、入力画像から決定論的にノイズを生成
- コンテンツ保持を強化
- 時間的に一貫したノイズにより、SW-CFAの効果を増幅

### アーキテクチャ
```
[Training]
Degraded Image → ControlNet → SD U-Net → L2 Loss with Clean Image
                    ↑
              Task Prompt (τ)

[Inference]
Video Frames → DDIM Inversion → Noisy Latents
                                    ↓
              ControlNet + SD U-Net (with SW-CFA) → Clean Latents → Restored Video
```

### 実験設定
- **学習データ**: 5つのタスク合計27,843枚の単一画像
  - Dehazing: REVIDE
  - Deraining: NTURain-syn
  - Denoising: DAVIS
  - MP4: MFQEv2
  - SR: REDS
- **テストデータ**: 各タスクの実世界ベンチマーク
- **ウィンドウサイズ**: N=3（前後各3フレーム）
- **サンプリング**: DDIM Inversion 10ステップ + DDIM backward 32ステップ

### 実験結果

#### 他手法との比較（平均FID/KID）
| Method | Type | FID↓ | KID↓ |
|--------|------|------|------|
| AirNet | Single-image regression | 89.47 | 5.21 |
| PromptIR | Single-image regression | 87.57 | 5.15 |
| VRT | Video regression | 89.23 | 5.17 |
| RVRT | Video regression | 90.57 | 5.20 |
| WeatherDiff | Single-image diffusion | 86.28 | 4.93 |
| InstructP2P | Single-image diffusion | 85.01 | 4.71 |
| **TDM (Ours)** | Video diffusion | **84.50** | **4.68** |

#### 時間的一貫性比較（Denoising）
| Method | FID↓ | FC↑ | WE↓ |
|--------|------|-----|-----|
| WeatherDiff | 80.11 | 9.439 | 2.182 |
| InstructP2P | 79.71 | 9.436 | 2.628 |
| **TDM** | **78.63** | 9.296 | **2.008** |

FC: Frame Consistency（高いほど良い）、WE: Warping Error（低いほど良い）

#### SW-CFAの効果
| Attention Type | Dehazing WE↓ | MP4 WE↓ |
|----------------|--------------|---------|
| 1st as Ref | 6.432 | 1.387 |
| Self-Attn (N=0) | 6.801 | 1.459 |
| **SW-CFA (N=3)** | **5.464** | **1.138** |

### アブレーションスタディ
| TPG | Inv. | SW-CFA | Dehazing FID↓ | Denoising FID↓ |
|-----|------|--------|---------------|----------------|
| ✓ | ✓ | - | 73.36 | 80.17 |
| ✓ | - | ✓ | 73.45 | 80.07 |
| - | ✓ | ✓ | 77.56 | 82.68 |
| ✓ | ✓ | ✓ | **73.68** | **78.63** |

- TPGなしでは画像品質（FID）が大幅に低下
- DDIM InversionとSW-CFAの組み合わせで時間的一貫性が向上

### 処理速度
- 15フレーム、512×896解像度: 30秒未満（RTX 4090）
- GPUメモリ: 10GB

### キーポイント
- 初のAll-in-One拡散ベースビデオ復元手法
- 単一画像データセットで学習し、ビデオ推論に適用可能
- TPGによるタスク切り替えは追加コストなし
- SW-CFAとDDIM Inversionの組み合わせで時間的一貫性を確保
- 実世界データへの汎化性能で既存手法を上回る
- 回帰ベース手法に比べ時間的一貫性でやや劣る（今後の課題）

---

## 9. Warped Diffusion (2410.16152)

**タイトル**: Warped Diffusion: Solving Video Inverse Problems with Image Diffusion Models

**著者**: Giannis Daras, Weili Nie, Karsten Kreis, Alexandros G. Dimakis, Morteza Mardani, Nikola B. Kovachki, Arash Vahdat (UT Austin, NVIDIA)

**発表**: NeurIPS 2024

### 概要
画像拡散モデルを使用してビデオ逆問題（インペインティング、超解像）を解決するフレームワーク。フレームを2D空間上の連続関数として捉え、ビデオをワーピング変換の連続として定式化。同変性自己誘導（Equivariance Self-Guidance）により時間的一貫性を確保。

### 課題
- 画像拡散モデルをビデオにフレーム単位で適用すると時間的一貫性が損なわれる
- ノイズワーピング手法（"How I Warped Your Noise"等）は拡散モデルが同変であることを前提とするが、実際には満たされない
- ホワイトノイズは連続的な評価ができない（分布であり、通常の関数ではない）
- 既存手法は潜在拡散モデル（Latent DM）で効果が限定的

### 理論的背景

#### 1. 関数空間における生成モデル
- 画像を連続関数 f: D → R³（D = [0,1]²）として定義
- ビデオをワーピング変換の系列として表現：
  ```
  f_j(x) = f_{j-1}(T_j^{-1}(x))  ∀x ∈ D ∩ D_j
  ```
- T_j はオプティカルフローで定義される変形写像

#### 2. 同変性の重要性
生成モデルGがワーピング変換T^{-1}に対して同変であることが必要：
```
G(ξ ∘ T^{-1})(x) = G(ξ)(T^{-1}(x))  ∀x ∈ D ∩ D_1
```
この条件が満たされない場合、ノイズワーピングのみでは時間的一貫性を達成できない。

### 提案手法: Warped Diffusion

#### 1. ガウス過程（GP）によるノイズ表現
ホワイトノイズの代わりにガウス過程を使用：
- 正定値カーネル関数 κ: R² × R² → R で定義
- 任意の点で連続的に評価可能
- Random Fourier Features (RFF) による効率的なサンプリング：
  ```
  ξ(x) = √(2/J) Σ_{j=1}^{J} w_j cos(⟨z_j, x⟩ + b_j)
  ```
  - w_j ~ N(0,1), z_j ~ N(0, ε^{-2}I_2), b_j ~ U(0, 2π)

#### 2. 関数空間拡散モデル
- 順方向過程: du_t = (2σ(t)σ̇(t)Q)^{1/2} dW_t
- 逆方向ODE: du_t/dt = -σ(t)σ̇(t) Q∇_u log p(u_t, t)
- 一般化されたTweedieの公式:
  ```
  Q∇_u log p(u_t, t) = (E[u_0|u_t] - u_t) / σ²(t)
  ```

#### 3. 同変性自己誘導（Equivariance Self-Guidance）
推論時にネットワークの同変性を強制：

**アルゴリズム**:
1. 最初のフレームを通常通り生成し、各タイムステップの出力を保存
2. 次フレームのノイズをRFFでワーピング: u_τ^{(j)} = u_τ^{(j-1)} ∘ T_j^{-1}
3. ODEサンプリング中、以下の誘導項を適用:
   ```
   e_t^{(j)} = |h_θ(u_t^{(j)}, t, c_j) ∘ T_j - h_θ(u_t^{(j-1)}, t, c_{j-1})|²
   u_{t-Δt}^{(j)} ← u_{t-Δt}^{(j)} - (λ/√e_t) ∇_u e_t^{(j)}
   ```

### 実装詳細
- **ベースモデル**: Stable Diffusion XL (SDXL)
- **ファインチューニング**: COYO データセットで100kステップ
- **GPパラメータ**:
  - RFF特徴数: 3000
  - 長さスケール: ε = 2/(π·resolution)
- **オプティカルフロー**: RAFT モデル
- **サンプリング**: 50ステップ（誘導あり）

### 実験結果

#### 単一フレーム評価
| Model | FID↓ | Inception↑ | CLIP Img↑ | SSIM↑ |
|-------|------|------------|-----------|-------|
| SR (Independent) | 40.84 | 11.68 | 0.957 | 0.785 |
| **SR (GP)** | **37.51** | **11.92** | 0.955 | 0.770 |
| Inpainting (Independent) | 61.38 | 11.71 | 0.913 | 0.778 |
| **Inpainting (GP)** | **58.73** | **11.77** | **0.929** | **0.798** |

GPノイズでの学習は独立ノイズと同等以上の性能。

#### インペインティングの時間的一貫性（2Dシフトタスク）
| Method | Warping Err↓ | FID↓ | CLIP Img↑ |
|--------|--------------|------|-----------|
| Fixed (gp) | 0.129 | 60.85 | 0.924 |
| Resample (gp) | 0.141 | 60.03 | 0.925 |
| How I Warped (indep) | 0.046 | 68.70 | 0.910 |
| GP Warping | 0.061 | 59.90 | 0.924 |
| **Warped Diffusion** | **0.001** | 61.25 | 0.917 |

Warped Diffusionはワーピングエラーをほぼゼロに削減。

#### 実ビデオでの8×超解像（FETVデータセット）
| Method | Warping Err↓ | FID↓ | CLIP Txt↑ | SSIM↑ |
|--------|--------------|------|-----------|-------|
| Fixed (indep) | 0.940 | 48.76 | 0.227 | 0.716 |
| Resample (indep) | 0.934 | 47.55 | 0.229 | 0.708 |
| GP Warping | 0.812 | 75.76 | 0.225 | 0.653 |
| **Warped Diffusion** | **0.649** | 58.19 | **0.235** | 0.654 |

時間的一貫性と復元品質のバランスを達成。

### ノイズワーピング速度
- GP Warping: 39ms/frame（1024×1024解像度）
- "How I Warped Your Noise"の16倍高速
- バッチ処理: 1000フレームを46msで生成可能

### 処理時間
- 誘導なし: 4.32 iterations/sec（A100 GPU）
- 誘導あり: 1.62 iterations/sec
- 2秒動画（16フレーム）: 約5分

### 制限事項
- 誘導項によりサンプリング時間が増加
- VAEデコーダが一部の変換で同変でない可能性（特にテキストレンダリング）
- オプティカルフロー推定の品質に依存
- オクルージョンがある場合に性能低下
- 相関ノイズで学習されたモデルが必要（ゼロショット不可）

### キーポイント
- 画像拡散モデルでビデオ逆問題を解決する初の体系的フレームワーク
- 関数空間の観点からノイズワーピングを理論的に定式化
- 同変性が時間的一貫性の鍵であることを特定
- 推論時誘導により追加学習なしで同変性を強制
- 潜在拡散モデル（SDXL等）に適用可能（従来手法の限界を克服）
- GPノイズでのファインチューニングは性能を損なわない
- ワーピングエラーをほぼゼロに削減しつつ高品質な復元を維持

---

## 10. TTVSR (2204.04216)

**タイトル**: Learning Trajectory-Aware Transformer for Video Super-Resolution

**著者**: Chengxu Liu, Huan Yang, Jianlong Fu, Xueming Qian (Xi'an Jiaotong University, Microsoft Research Asia)

**発表**: CVPR 2022

### 概要
Video Super-Resolution（VSR）にTransformerアーキテクチャを効果的に導入した初期の重要な研究。時空間軌跡（Trajectory）に沿ったアテンション計算により、計算コストを大幅に削減しながら長距離フレーム依存性のモデリングを実現。

### 課題
- **スライディングウィンドウ手法**: 隣接5-7フレームのみ利用、遠方フレームのテクスチャを活用できない
- **リカレント手法**: 勾配消失問題により長期モデリング能力が制限される
- **Vanilla Vision Transformer**: 動画への適用は計算コストが膨大

### 提案手法

#### 1. Trajectory-aware Attention（軌跡認識アテンション）
動画フレームを事前整列された視覚トークンの軌跡として定式化：

```
軌跡の定義:
T = {τ_i, i ∈ [1,N]}
τ_i = ⟨τ_i^t = (x_i^t, y_i^t), t ∈ [1,T]⟩

アテンション計算:
h_τi = argmax_t ⟨q_τi^T / ||q_τi^T||₂², k_τi^t / ||k_τi^t||₂²⟩  (Hard Attention)
s_τi = max_t ⟨q_τi^T / ||q_τi^T||₂², k_τi^t / ||k_τi^t||₂²⟩    (Soft Attention)

A_traj(q, k, v) = C(q_τi^T, s_τi ⊙ v_τi^{h_τi})
```

**Hard Attention**: 軌跡上で最も関連性の高いトークンを選択（重み付き平均によるぼやけを防止）
**Soft Attention**: 関連パッチの信頼度を生成（不正確な場合の影響を軽減）

#### 2. Location Maps（位置マップ）
軌跡生成を効率的な行列演算で実現：

```
L^t = [(x_1,y_1) ... (x_1,y_W)]
      [   ...    ...    ...  ]
      [(x_H,y_1) ... (x_H,y_W)], t ∈ [1,T]

位置マップ更新:
*L^t = S(L^t, O^{T+1})
```

- O^{T+1}: 軽量動き推定ネットワーク（SPyNet）からのバックワードフロー
- S(·): 空間サンプリング操作（grid_sample）
- 軌跡をオンラインで効率的に更新可能

#### 3. Cross-scale Feature Tokenization（CFT）
長距離動画で頻発するスケール変化問題への対処：

```
処理フロー:
1. Unfold/Fold操作で受容野を拡大
2. Pooling操作で異なるスケールの特徴を同一サイズに統一
3. Unfold操作でトークンを抽出

トークンサイズ: 4×4, 6×6, 8×8（3スケール）
```

大きなスケールのテクスチャを小さなスケールの復元に活用。

### アーキテクチャ詳細

```
全体構成:
┌─────────────────────────────────────────────────────────────────┐
│ I_LR^T → ϕ(·) → Q                                               │
│                                                                 │
│ I_LR → H(·) → L (Location Maps)                                │
│      → ϕ(·) → K                                                │
│      → φ(·) → V                                                │
│                                                                 │
│ Trajectory-aware Attention → R(·) → Pixel Shuffle → + → I_SR^T │
│                                            ↑                    │
│                            U(I_LR^T) ──────┘                    │
└─────────────────────────────────────────────────────────────────┘

ネットワーク構成:
- 特徴抽出: 5 Residual Blocks（64チャネル）
- 再構成ネットワーク: 60 Residual Blocks
- 双方向伝播スキーム採用
- 動き推定: 事前学習済みSPyNet
```

### 計算量削減

Vanilla Vision TransformerとTTVSRの比較：

```
Vanilla Attention計算量:
(T × H/D_h × W/D_w) × (C × D_h × D_w)

Trajectory-aware Attention計算量:
(T × 1 × 1) × (C × D_h × D_w)

削減率: (H/D_h × W/D_w) 倍
```

空間次元のアテンション計算を回避し、計算量を大幅削減。

### 訓練設定
- **オプティマイザ**: Adam (β₁=0.9, β₂=0.99)
- **学習率**: 動き推定 1.25×10⁻⁵, その他 2×10⁻⁴
- **バッチサイズ**: 8
- **入力パッチ**: 64×64
- **シーケンス長**: 50フレーム（REDS）
- **損失関数**: Charbonnier penalty loss
- **イテレーション**: 400K（最初の5Kで動き推定は固定）
- **データ拡張**: ランダム水平/垂直反転、90°回転

### 実験結果

#### REDS4データセット（4× SR, RGB）
| Method | #Frame | Clip 000 | Clip 011 | Clip 015 | Clip 020 | Average |
|--------|--------|----------|----------|----------|----------|---------|
| EDVR | 7 | 28.01/0.8250 | 32.17/0.8864 | 34.06/0.9206 | 30.09/0.8881 | 31.09/0.8800 |
| MuCAN | 5 | 27.99/0.8219 | 31.84/0.8801 | 33.90/0.9170 | 29.78/0.8811 | 30.88/0.8750 |
| BasicVSR | r | 28.39/0.8429 | 32.46/0.8975 | 34.22/0.9237 | 30.60/0.8996 | 31.42/0.8909 |
| IconVSR | r | 28.55/0.8478 | 32.89/0.9024 | 34.54/0.9270 | 30.80/0.9033 | 31.67/0.8948 |
| **TTVSR** | r | **28.82/0.8566** | **33.47/0.9100** | **35.01/0.9325** | **31.17/0.9094** | **32.12/0.9021** |

IconVSRを**0.45dB上回る**大幅な改善。

#### その他のデータセット（4× SR, Y-channel）
| Method | Vid4 | UDM10 | Vimeo-90K-T |
|--------|------|-------|-------------|
| EDVR | 27.85/0.8503 | 39.89/0.9686 | 37.81/0.9523 |
| BasicVSR | 27.96/0.8553 | 39.96/0.9694 | 37.53/0.9498 |
| IconVSR | 28.04/0.8570 | 40.03/0.9694 | 37.84/0.9524 |
| **TTVSR** | **28.40/0.8643** | **40.41/0.9712** | **37.92/0.9526** |

長いシーケンス（30フレーム以上）で特に大きな改善。

#### モデルサイズと計算コスト
| Method | #Params(M) | FLOPs(T) | PSNR/SSIM |
|--------|------------|----------|-----------|
| DUF | 5.8 | 2.34 | 28.63/0.8251 |
| EDVR | 20.6 | 2.95 | 31.09/0.8800 |
| MuCAN | 13.6 | >1.07 | 30.88/0.8750 |
| BasicVSR | 6.3 | 0.33 | 31.42/0.8909 |
| IconVSR | 8.7 | 0.51 | 31.67/0.8948 |
| **TTVSR** | **6.8** | **0.61** | **32.12/0.9021** |

アテンションベース手法（MuCAN）と比較して軽量かつ高性能。

### アブレーション実験

#### 軌跡認識アテンションの効果
| Method | TG | TA | PSNR/SSIM |
|--------|----|----|-----------|
| Base | - | - | 30.46/0.8661 |
| Base+TG | ✓ | - | 31.91/0.8985 |
| Base+TG+TA | ✓ | ✓ | 31.99/0.9007 |

軌跡生成（TG）で+1.45dB、軌跡アテンション（TA）でさらに+0.08dB改善。

#### フレーム数の影響
| #Frame | 5 | 10 | 20 | 33 | 45 |
|--------|---|----|----|----|----|
| PSNR | 31.89 | 31.93 | 31.97 | 31.99 | 32.01 |

フレーム数と性能は正の相関。33フレーム（3フレーム間隔）で十分。

#### Cross-scale Feature Tokenizationの効果
| Method | Token sizes | PSNR/SSIM |
|--------|-------------|-----------|
| Base+TG+TA | 4 | 31.99/0.9007 |
| +CFT(S2) | 4, 6 | 32.08/0.9011 |
| +CFT(S3) | 4, 6, 8 | **32.12/0.9021** |

マルチスケールトークン化で+0.13dB改善。

### 制限事項
- **回転への対応**: 回転が発生すると軌跡が不正確になり性能低下
- **短いシーケンス**: Vimeo-90K（7フレーム）では改善が限定的
- **訓練時間**: 長シーケンス入力により訓練時間が増加
- **GPUメモリ**: 各時刻の特徴を保存する必要あり

### キーポイント
- VSRにTransformerを効果的に導入した先駆的研究（CVPR 2022）
- 軌跡ベースのアテンションで計算量を(H/D_h × W/D_w)倍削減
- 長距離フレーム依存性のモデリングを実現（50フレーム以上対応）
- IconVSRを0.45dB上回るSOTA性能（REDS4）
- 軽量かつ効率的（6.8Mパラメータ、0.61T FLOPs）
- Cross-scale Feature Tokenizationでスケール変化に対応
- 拡散モデルを使用しない従来型深層学習アプローチの重要なベースライン

---

## 全体まとめ

### 論文一覧

| No. | 論文名 | 発表 | 手法タイプ | 主要貢献 |
|-----|--------|------|------------|----------|
| 1 | UltraVSR | MM 2025 | 1ステップ拡散 | 劣化度に応じた1ステップ再構成、RTS時間整合性 |
| 2 | DiTVR | arXiv 2025 | DiT拡散 | 時空間分離アテンション、カスケード推論 |
| 3 | VDM-VSR | ICASSP 2025 | Video Diffusion | 3D VAE、時間的拡散で一貫性確保 |
| 4 | SDATC | arXiv 2025 | Stable Diffusion | 適応的タイルベース処理、TAWFMフロー融合 |
| 5 | FCVSR | arXiv 2025 | Stable Diffusion | フロー連結表現、時間一貫性蒸留 |
| 6 | RCDM | arXiv 2025 | Rectified Flow | 正規化フローで軌跡直線化、高速サンプリング |
| 7 | RepNet-VSR | arXiv 2025 | 構造再パラメータ化 | 推論時のブロック統合、効率的設計 |
| 8 | TDM | arXiv 2025 | ControlNet拡散 | マルチタスク復元、TPGプロンプト誘導 |
| 9 | Warped Diffusion | NeurIPS 2024 | SDXL + GP | 関数空間ノイズ、同変性誘導 |
| 10 | TTVSR | CVPR 2022 | Transformer | 軌跡アテンション、長距離依存性モデリング |

### 技術的トレンド

#### 1. 拡散モデルの台頭
2022年以降、VSR分野で拡散モデル（Diffusion Model）が主流に：
- **Stable Diffusion系**: SDATC, FCVSR, TDM など事前学習済みSDを活用
- **Video Diffusion系**: VDM-VSR は3D VAEで時間軸も含めた生成
- **DiT系**: DiTVR はTransformerベースの拡散アーキテクチャを採用
- **1ステップ化**: UltraVSR, RCDM で高速化を追求

#### 2. 時間的一貫性の確保手法
フレーム間の整合性確保が最重要課題：
- **アテンションベース**: Cross-Frame Attention (SW-CFA, TCA)
- **フローベース**: オプティカルフローによるワーピング (SDATC, FCVSR, Warped Diffusion)
- **特徴シフト**: RTS, Temporal Shift (UltraVSR, RepNet-VSR)
- **軌跡ベース**: Trajectory-aware Attention (TTVSR)
- **蒸留ベース**: Temporal Consistency Distillation (FCVSR, UltraVSR)

#### 3. 効率化の取り組み
推論速度向上のための工夫：
- **ステップ削減**: 1ステップ拡散 (UltraVSR)、Rectified Flow (RCDM)
- **カスケード処理**: 低解像度→高解像度の段階的処理 (DiTVR)
- **タイル処理**: 大きな動画を分割処理 (SDATC)
- **構造再パラメータ化**: 推論時のモデル簡略化 (RepNet-VSR)
- **軌跡計算効率化**: Location Maps (TTVSR)

#### 4. ベースモデルの選択
- **Stable Diffusion 2.1/XL**: SDATC, FCVSR, Warped Diffusion
- **ControlNet**: TDM
- **DiT**: DiTVR
- **カスタムUNet**: VDM-VSR, RCDM
- **非拡散**: TTVSR (Transformer), RepNet-VSR (CNN)

### 性能比較（REDS4, 4× SR）

| Method | PSNR↑ | 推論速度 | 備考 |
|--------|-------|---------|------|
| TTVSR (2022) | 32.12 | 高速 | Transformerベースライン |
| DiTVR (2025) | - | 中速 | DiTベース、品質重視 |
| UltraVSR (2025) | - | 最高速 | 1ステップ、720p: 0.89秒 |
| RCDM (2025) | 28.10 | 高速 | 1ステップ、Rectified Flow |
| VDM-VSR (2025) | 28.70* | 低速 | 高品質、時間一貫性優秀 |

*Vid4データセットでの値

### 主要な課題と今後の方向性

1. **速度と品質のトレードオフ**: 1ステップ化により速度は改善したが、マルチステップに比べ品質が低下する傾向
2. **オクルージョン対応**: 遮蔽領域での時間的一貫性維持が困難
3. **回転・大きな動きへの対応**: フローベース手法の限界
4. **Real-world劣化への対応**: 合成データと実データのギャップ
5. **メモリ効率**: 長時間動画処理時のGPUメモリ制約

### 結論

VSR分野は拡散モデルの導入により大きく進展し、2024-2025年にかけて多くの革新的手法が提案された。特に：
- **時間的一貫性**と**推論効率**の両立が主要な研究テーマ
- Stable Diffusionなどの事前学習モデルの活用が標準化
- 1ステップ拡散やRectified Flowによる高速化が進展
- フローベースワーピングとアテンションベース融合の併用が効果的

TTVSRのような従来型Transformer手法も依然として効率性の観点で重要なベースラインとして位置づけられている。

