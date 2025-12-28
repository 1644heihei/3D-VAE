import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PairedMRIDataset(Dataset):
    def __init__(self, root_dir="dataset", mode="train", transform=None):
        """
        Args:
            root_dir (string): データセットのルートディレクトリ (例: "dataset")
            mode (string): 'train', 'test' のいずれか
            transform (callable, optional): サンプルに適用される変換
        """
        self.lr_dir = os.path.join(root_dir, mode, "LR")
        self.hr_dir = os.path.join(root_dir, mode, "HR")

        if not os.path.exists(self.lr_dir):
            raise FileNotFoundError(
                f"Directory not found: {self.lr_dir}. Please run prepare_data.py first."
            )

        self.filenames = [f for f in os.listdir(self.lr_dir) if f.endswith(".raw")]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        lr_path = os.path.join(self.lr_dir, filename)
        hr_path = os.path.join(self.hr_dir, filename)

        # .rawファイルを読み込み (int16)
        # サイズは prepare_data.py で指定した (80, 96, 80) を想定
        # 必要に応じて shape を変更してください
        shape = (80, 96, 80)

        lr_image = np.fromfile(lr_path, dtype=np.int16).reshape(shape)
        hr_image = np.fromfile(hr_path, dtype=np.int16).reshape(shape)

        # 正規化 (0-1範囲に)
        # データの最大値が不明な場合は、データセット全体の統計を使うか、
        # ここでは簡易的に画像ごとの min-max 正規化、あるいは固定値 (例: 255や4095) で割る方法があります。
        # ここでは prepare_data.py で np.clip(0, 255) されていると仮定して 255 で割ります。
        # もし元データが CT値などで範囲が広い場合は調整が必要です。

        lr_image = lr_image.astype(np.float32) / 255.0
        hr_image = hr_image.astype(np.float32) / 255.0

        # チャンネル次元を追加 (C, D, H, W) -> (1, 80, 96, 80)
        lr_image = torch.from_numpy(lr_image).unsqueeze(0)
        hr_image = torch.from_numpy(hr_image).unsqueeze(0)

        if self.transform:
            # 必要であれば transform を適用 (回転や反転など)
            pass

        return lr_image, hr_image


def get_dataloader(
    root_dir="dataset", mode="train", batch_size=4, shuffle=True, num_workers=2
):
    dataset = PairedMRIDataset(root_dir, mode=mode)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    # テスト用コード
    root = "dataset"
    if os.path.exists(root):
        try:
            loader = get_dataloader(root, mode="train", batch_size=2)
            print("Checking Train Loader...")
            for lr, hr in loader:
                print(f"LR shape: {lr.shape}, HR shape: {hr.shape}")
                break
        except Exception as e:
            print(e)
    else:
        print(f"Directory {root} not found. Please run prepare_data.py first.")
