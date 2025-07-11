from model import *
from loss import compound_transunet_loss
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

LOSS_EXPONENTS_BETA = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
LOSS_EXPONENTS_GAMMA = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

def train(model: TransUNet3plus,
          data_train,
          data_test=None,
          batch_size=8,
          epochs=10,
          early_stop_threshold=1e-4,
          early_stop_patience=5,
          input_image_size=256,
          save_model_directory=None,
          device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    dl_train = DataLoader(data_train, batch_size, shuffle=True, pin_memory=False)
    dl_test = DataLoader(data_test, batch_size, pin_memory=False) if data_test else None

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    resizer = nn.Upsample((input_image_size, input_image_size))

    best_test_loss = float('inf')
    best_model_path = None
    patience = early_stop_patience

    for epoch in range(epochs):
        print(f"\nEPOCH {epoch+1} / {epochs} :")
        model.train()
        train_loss_total = 0.0
        train_loss_mask = 0.0

        # 训练循环
        with tqdm(dl_train, desc=f"Epoch {epoch+1}/{epochs}") as train_batches:
            for i, (image, mask) in enumerate(train_batches):
                optimizer.zero_grad()
                image = image.to(device)
                mask = mask.to(device)
                im_resized = resizer(image)
                mask_resized = resizer(mask)
                mask_output = model(im_resized)

                preds = []
                for layer in model.sequp[:-1]:
                    if hasattr(layer, 'side_mask_output') and layer.side_mask_output is not None:
                        pred = resizer(layer.side_mask_output)
                        if pred.shape[1] > 1:
                            pred = pred[:, 1:, ...]
                        preds.append(pred)
                final_pred = mask_output
                if final_pred.shape[1] > 1:
                    final_pred = final_pred[:, 1:, ...]
                preds.append(final_pred)
                target = mask_resized[:, 1:, ...] if mask_resized.shape[1] > 1 else mask_resized

                image_loss = compound_transunet_loss(preds, target, LOSS_EXPONENTS_BETA, LOSS_EXPONENTS_GAMMA)
                total_loss = image_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_total += total_loss.item()
                train_loss_mask += image_loss.item()
                train_batches.set_postfix({
                    'Total Loss': f"{train_loss_total/(i+1):.5f}",
                    'Seg Loss': f"{train_loss_mask/(i+1):.5f}"
                })

        # 测试循环
        if dl_test:
            model.eval()
            test_loss_total = 0.0
            test_loss_mask = 0.0
            with torch.no_grad():
                with tqdm(dl_test, desc="Testing") as test_batches:
                    for i, batch in enumerate(test_batches):
                        # 兼容测试集只有image的情况
                        if isinstance(batch, (list, tuple)) and len(batch) == 2:
                            image, mask = batch
                            mask = mask.to(device)
                        else:
                            image = batch
                            mask = None
                        image = image.to(device)
                        im_resized = resizer(image)
                        mask_output = model(im_resized)
                        preds = []
                        for layer in model.sequp[:-1]:
                            if hasattr(layer, 'side_mask_output') and layer.side_mask_output is not None:
                                pred = resizer(layer.side_mask_output)
                                if pred.shape[1] > 1:
                                    pred = pred[:, 1:, ...]
                                preds.append(pred)
                        final_pred = mask_output
                        if final_pred.shape[1] > 1:
                            final_pred = final_pred[:, 1:, ...]
                        preds.append(final_pred)
                        if mask is not None:
                            mask_resized = resizer(mask)
                            target = mask_resized[:, 1:, ...] if mask_resized.shape[1] > 1 else mask_resized
                            image_loss = compound_transunet_loss(preds, target, LOSS_EXPONENTS_BETA, LOSS_EXPONENTS_GAMMA)
                            total_loss = image_loss
                            test_loss_total += total_loss.item()
                            test_loss_mask += image_loss.item()
                            test_batches.set_postfix({
                                'Total Loss': f"{test_loss_total/(i+1):.5f}",
                                'Seg Loss': f"{test_loss_mask/(i+1):.5f}"
                            })
            avg_test_loss = test_loss_total / len(dl_test)
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                if save_model_directory:
                    os.makedirs(save_model_directory, exist_ok=True)
                    best_model_path = os.path.join(save_model_directory, "best_model.pth")
                    torch.save(model.state_dict(), best_model_path)
                    print(f"保存最优模型权重于: {best_model_path} (Test Loss: {avg_test_loss:.5f})")
    
    # 保存最终模型
    if save_model_directory:
        os.makedirs(save_model_directory, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_model_directory, "final_model.pth"))