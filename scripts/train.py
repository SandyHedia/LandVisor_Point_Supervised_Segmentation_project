import torch
import yaml
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

# Import custom modules
from src import LandVisorTrainer,LandVisorModel, apply_histogram_matching,PartialFocalCELoss, RemoteSensingDataset, AugmentationFactory

def main():
    
    # 1. Load the Main Config
    with open("configs/base_config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)

    with open("configs\model\model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)

    config = base_cfg
    config['model'] = model_cfg

    device = torch.device(config['training']['device']) #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = config['training']['epochs']
    
    # 2. Setup Dataloaders using config values
    train_ds = RemoteSensingDataset(
        image_dir=config['data']['train_images'],
        mask_dir=config['data']['train_masks'],
        reference_path=config['data']['reference_path'],
        points_per_class=config['data']['points_per_class'],
        transform=AugmentationFactory.get_train_transform(config['data']['image_size'])
    )

    val_ds = RemoteSensingDataset(
        image_dir=config['data']['val_images'],
        mask_dir=config['data']['val_masks'],
        reference_path=config['data']['reference_path'],
        points_per_class=config['data']['points_per_class'],
        transform=AugmentationFactory.get_val_transform(config['data']['image_size'])
    )
    

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False)

    
    # 3. Model Initialization 
    student = LandVisorModel(config, pretrained=True)
    

    # 4. Initialize the Trainer
    trainer = LandVisorTrainer(
        student_model=student,
        dataloader_train=train_loader,
        dataloader_val=val_loader,
        criterion_pfce=PartialFocalCELoss(gamma=config['training']['gamma']),
        optimizer=torch.optim.Adam(student.parameters(), lr=float(config['training']['learning_rate'])),
        device=device,
        ema_alpha=config['training']['ema_alpha'],
        consistency_weight=config['training']['consistency_weight']
    )
    
    # 6. Training Loop
    print(f"Starting training on {device}...")
    for epoch in range(1, EPOCHS + 1):
        train_metrics = trainer.train_epoch(epoch)
        
        # Every 5 epochs, run a full validation
        if epoch % 5 == 0:
            miou, class_mious = trainer.validate()
            print(f"Epoch {epoch} | mIoU: {miou:.4f} | Loss: {train_metrics['loss']:.4f}")
            
    # 7. Save the Final Teacher Model (The most stable one)
    torch.save(trainer.teacher.state_dict(), "landvisor_final_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
