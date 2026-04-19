import torch
import yaml
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src import LandVisorModel, RemoteSensingDataset, visualize_results, AugmentationFactory

def evaluate():
    # 1. Load the Main Config
    with open("configs/base_config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)

    with open("configs/model/unet_resnet34.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)

    config = base_cfg
    config['model'] = model_cfg

    device = torch.device(config['training']['device'])
    MODEL_PATH = config['evaluation']['model_path']
    REFERENCE_IMG = config['evaluation']['reference_img']
    NUM_CLASSES = config['data']['num_classes']
    CLASS_NAMES = config['data']['class_names']

    # 2. Load Data
    test_imgs, test_masks = config['data']['test_images'], config['data']['test_masks']
    test_ds = RemoteSensingDataset(
    image_dir=test_imgs, 
    mask_dir=test_masks, 
    transform=AugmentationFactory.get_val_transform(config['data']['image_size']),
    reference_path=REFERENCE_IMG
)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    # 3. Load the Trained Teacher Model
    model = LandVisorModel(config, pretrained=False) # No need for pretrained weights here
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    print(f"Starting Evaluation on {len(test_ds)} images...")

    # 4. Run Evaluation
    # We reuse the logic from our trainer to get mIoU and per-class IoU
    all_mious = []
    per_class_accum = {i: [] for i in range(NUM_CLASSES)}

    with torch.no_grad():
        for i, (images, sparse_masks, full_masks) in enumerate(test_loader):
            images = images.to(device)
            full_masks = full_masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)


            for cls in range(NUM_CLASSES):
                inter = ((preds == cls) & (full_masks == cls)).sum().float()
                union = ((preds == cls) | (full_masks == cls)).sum().float()
                if union > 0:
                    iou = (inter / union).item()
                    per_class_accum[cls].append(iou)

            # 5. Save a "Showcase" Visualization for the first batch
            if i == 0:
                visualize_results(
                    image=images[0].permute(1,2,0).cpu().numpy().astype(np.uint8),
                    sparse_mask=sparse_masks[0].cpu().numpy(),
                    prediction=preds[0].cpu().numpy(),
                    full_mask=full_masks[0].cpu().numpy(),
                    save_path="evaluation_showcase.png"
                )

    # 6. Aggregate Results into a Professional Table
    report_data = []
    for cls_idx, name in enumerate(CLASS_NAMES):
        avg_iou = np.mean(per_class_accum[cls_idx]) if per_class_accum[cls_idx] else 0
        report_data.append({"Class": name, "IoU": round(avg_iou, 4)})
    
    df = pd.DataFrame(report_data)
    mean_iou = df["IoU"].mean()
    
    print("\n--- FINAL PERFORMANCE REPORT ---")
    print(df.to_string(index=False))
    print(f"\nOverall Mean IoU (mIoU): {mean_iou:.4f}")
    
    # Save report to CSV for the technical document
    df.to_csv("test_results_report.csv", index=False)

if __name__ == "__main__":
    evaluate()
