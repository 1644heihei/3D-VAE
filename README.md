# 3D Latent Diffusion Model (LDM) 画質改善プロジェクト

このプロジェクトは、3Dの医用画像（血管画像など）の画質を良くするAIを作るためのプログラム一式です。
「3D-VAE」と「Diffusion Model」という2つの最新技術を組み合わせて、ノイズを除去し、くっきりとした画像を作ります。

## 全体の流れ

このAIを作るには、以下の5つのステップを順番に実行する必要があります。

1.  **データ準備**: 大きな画像を、AIが学習しやすいサイズに切り分ける。
2.  **Phase 1 (VAE)**: 画像を「特徴（潜在変数）」に圧縮するAIを育てる。
3.  **Phase 2 (変換)**: 育てたVAEを使って、全画像を「特徴」に変換する。
4.  **Phase 3 (Diffusion)**: 「特徴」をきれいにするAIを育てる。
5.  **Phase 4 (推論)**: 完成したAIを使って、実際に画像をきれいにする。

---

## 準備 (Environment)

以下のライブラリが必要です。
*   Python 3.8以上
*   PyTorch (GPU版推奨)
*   NumPy, Pillow, tqdm

---

## 実行手順 (Step-by-Step)

### Step 0: データの準備
まずは元データを学習用(`train`)とテスト用(`test`)に分けて、小さなブロックに切り出します。

*   **実行コマンド**:
    ```powershell
    python prepare_data.py
    ```
*   **何が起きる？**: `dataset/train` と `dataset/test` フォルダに、`.raw` ファイルがたくさん作られます。

### Step 1: VAEの学習 (Phase 1)
画像を圧縮・復元する「土台」となるAI (VAE) を学習させます。

*   **実行コマンド**:
    ```powershell
    python train_vae_pair.py
    ```
*   **時間の目安**: 数時間〜半日（GPU性能による）
*   **確認**: `checkpoint/vae_denoise/` にモデルファイル (`.pt`) が保存されます。

### Step 2: VAEの画質チェック
VAEが正しく学習できたか確認します。

*   **実行コマンド**:
    ```powershell
    python evaluate_psnr.py
    ```
*   **判定基準**: 画面に表示される `Result PSNR` が `Baseline PSNR` より大きければOKです。

### Step 3: 潜在変数への変換 (Phase 2)
学習したVAEを使って、重たい画像データを軽量な「潜在変数」データに変換します。これを行うと、次の学習が劇的に速くなります。

*   **実行コマンド**:
    ```powershell
    python prepare_latents.py
    ```
*   **何が起きる？**: `dataset/latents` フォルダに `.pt` ファイルが保存されます。

### Step 4: Diffusion Modelの学習 (Phase 3)
ここが本番です。ボケた特徴をくっきりさせる「Diffusion Model」を学習させます。

*   **実行コマンド**:
    ```powershell
    python train_diffusion.py
    ```
*   **時間の目安**: 数時間
*   **確認**: `checkpoint/diffusion/` にモデルファイルが保存されます。

### Step 5: 最終結果の確認 (Phase 4)
完成した2つのモデルを組み合わせて、実際に画質改善を行います。

*   **実行コマンド**:
    ```powershell
    python predict_ldm.py
    ```
*   **結果**: `results/ldm_prediction/` フォルダに画像 (`.png`) が保存されます。
    *   `1_input_LR.png`: 元の低画質画像
    *   `2_target_HR.png`: 正解の高画質画像
    *   `3_output_LDM.png`: **AIが画質改善した画像**

---

## ファイル一覧 (Files)

*   `prepare_data.py`: データ前処理用
*   `train_vae_pair.py`: VAE学習用
*   `evaluate_psnr.py`: 画質評価用
*   `prepare_latents.py`: 潜在変数変換用
*   `train_diffusion.py`: Diffusion学習用
*   `predict_ldm.py`: 推論・画像生成用
*   `model_vae_3d.py`: VAEの設計図
*   `model_diffusion.py`: Diffusionの設計図
*   `dataloader_pair.py`: データ読み込み処理

## トラブルシューティング

*   **"CUDA out of memory" エラーが出る**:
    *   `train_vae_pair.py` や `train_diffusion.py` の `batch_size` を小さくしてください（例: 4 -> 2）。
*   **画質が良くならない**:
    *   Step 2の時点で画質が悪い場合、VAEの学習不足です。Epoch数を増やしてください。
