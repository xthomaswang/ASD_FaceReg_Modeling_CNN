# src/models/cornet_wrapper.py

import torch
import torch.nn as nn
import cornet
from torchvision import transforms

class CornetWithPathology(nn.Module):
    """
    A wrapper around CORnet (Z/S) that injects ASD-like pathology
    (E/I Imbalance and Internal Noise) without retraining the weights.
    """
    def __init__(self, model_name='Z', slope=1.0, noise_std=0.0, device='cuda'):
        super().__init__()
        self.device = device
        self.slope = slope       # Corresponds to slope_positive (E/I gain)
        self.noise_std = noise_std
        self.model_name = model_name

        print(f"Loading CORnet-{model_name} (Pre-trained on ImageNet)...")
        
        # 1. Load pretrained model
        # map_location ensures direct loading to GPU or CPU
        if model_name == 'Z':
            self.core_model = cornet.cornet_z(pretrained=True, map_location=device)
        elif model_name == 'S':
            self.core_model = cornet.cornet_s(pretrained=True, map_location=device)
        elif model_name == 'RT':
            self.core_model = cornet.cornet_rt(pretrained=True, map_location=device)
        else:
            raise ValueError("Model must be Z, S, or RT")

        # Unwrap once (works for DataParallel or not)
        self.base = self.core_model.module if isinstance(self.core_model, nn.DataParallel) else self.core_model

        # 2. Freeze all parameters (Freeze Backbone)
        # Key for reviewers: No retraining - we want to see pretrained brain's response under pathology
        for param in self.core_model.parameters():
            param.requires_grad = False
        
        self.core_model.eval() # Critical: Lock BatchNorm and Dropout
        self.core_model.to(device)

        # 3. Register Hooks (Key step for injecting E/I and Noise)
        self.hooks = []
        self._register_pathology_hooks()

    def _pathology_hook(self, module, input, output):
        """
        This function automatically executes after forward propagation of each layer.
        Output is the raw output features of that layer (e.g., V1, V2...).
        """
        # A. Simulate E/I Imbalance (Gain Modulation)
        # If slope > 1.0, simulate E > I (ASD)
        # If slope < 1.0, simulate I > E
        modulated_output = output * self.slope

        # B. Simulate Internal Noise
        # Inject Gaussian noise into neural activity
        if self.noise_std > 0:
            noise = torch.randn_like(modulated_output) * self.noise_std
            modulated_output = modulated_output + noise

        return modulated_output

    def _register_pathology_hooks(self):
        """
        Attach hooks to nonlinearity outputs (more faithful E/I manipulation)
        Hooks inject pathology AFTER nonlinear activation, simulating altered E/I balance
        """
        # CORblock_Z has .nonlin attribute for nonlinearity
        target_modules = [
            self.base.V1.nonlin,
            self.base.V2.nonlin,
            self.base.V4.nonlin,
            self.base.IT.nonlin,
        ]
        
        for mod in target_modules:
            # register_forward_hook allows us to modify outputs
            h = mod.register_forward_hook(self._pathology_hook)
            self.hooks.append(h)
        
        print(f"Pathology Injected (nonlin hooks): Slope={self.slope}, Noise={self.noise_std}")

    def forward(self, x):
        return self.core_model(x)

    def close(self):
        """Remove hooks and clean up memory"""
        for h in self.hooks:
            h.remove()

# ==========================================
# Utility Functions: Data Preprocessing
# ==========================================
def get_cornet_transforms():
    """
    CORnet requires standard ImageNet preprocessing
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet standard mean
                             std=[0.229, 0.224, 0.225])  # ImageNet standard std
    ])

# ==========================================
# Main Function (Corresponds to original build_cnn)
# ==========================================
def extract_features(data_tensor, model_name='Z', slope=1.0, noise_level=0.0, device='cuda'):
    """
    Extract IT-layer features from a CORnet model with pathology injection.

    Uses a forward hook on the IT area to capture internal representations
    rather than the final decoder logits.

    Inputs:
        data_tensor: PyTorch Tensor [Batch, 3, Height, Width] (already normalized)
        model_name: 'Z' or 'S'
        slope: E/I ratio
        noise_level: Noise std

    Outputs:
        features: numpy array of IT layer output (flattened)
    """
    # 1. Build model
    model = CornetWithPathology(model_name, slope=slope, noise_std=noise_level, device=device)

    # 2. Register a hook on the IT area to capture its output
    it_features = {}

    def _capture_it(module, input, output):
        it_features['out'] = output.detach()

    hook = model.base.IT.register_forward_hook(_capture_it)

    # 3. Inference — run the full forward pass so pathology hooks also fire
    with torch.no_grad():
        if device == 'cuda':
            data_tensor = data_tensor.cuda()
        model(data_tensor)

    hook.remove()

    # 4. Flatten spatial dims: (B, C, H, W) -> (B, C*H*W)
    feat = it_features['out'].cpu()
    if feat.dim() > 2:
        feat = feat.flatten(1)

    # Cleanup
    model.close()

    return feat.numpy()


# ==========================================
# Post-Training Functions
# ==========================================

def build_cornet_for_training(
    num_classes, 
    alpha=1.0, 
    noise_std=0.0, 
    freeze_backbone=False, 
    pretrained=True,
    penultimate_dim=64,
    penultimate_dropout=0.5
):
    """
    Build CORnet for post-training with custom activation and penultimate dense head.
    
    Parameters:
        num_classes: number of output classes
        alpha: E/I gain modulation
        noise_std: internal noise level
        freeze_backbone: if True, only train decoder
        pretrained: use ImageNet pretrained weights
        penultimate_dim: dimension of penultimate dense layer (default 64, like Keras CNN)
        penultimate_dropout: dropout rate for penultimate layer
        
    Returns:
        Modified CORnet model ready for training
    """
    import cornet
    from collections import OrderedDict
    from .custom_layers import EIRectifiedLinear
    
    print(f"Building CORnet for training: alpha={alpha}, noise={noise_std}")
    
    # Load pretrained CORnet-Z
    model = cornet.cornet_z(pretrained=pretrained)
    
    # Handle DataParallel wrapper robustly
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    
    # Replace all ReLU and existing EIRectifiedLinear with updated alpha/noise
    def replace_nonlin(module):
        for name, child in module.named_children():
            # Replace ReLU
            if isinstance(child, nn.ReLU):
                setattr(module, name, EIRectifiedLinear(alpha, noise_std))
            # Also replace existing EIRectifiedLinear to ensure alpha/noise take effect
            elif child.__class__.__name__ == "EIRectifiedLinear":
                setattr(module, name, EIRectifiedLinear(alpha, noise_std))
            else:
                replace_nonlin(child)
    
    replace_nonlin(actual_model)
    
    # Rebuild decoder
    in_features = actual_model.decoder.linear.in_features  # 512
    old_avgpool = actual_model.decoder.avgpool
    old_flatten = actual_model.decoder.flatten
    
    if penultimate_dim == 0:
        # Use original architecture: 512 -> num_classes 
        actual_model.decoder = nn.Sequential(OrderedDict([
            ("avgpool", old_avgpool),
            ("flatten", old_flatten),
            ("linear", nn.Linear(in_features, num_classes)),
            ("output", nn.Identity()),
        ]))
        print(f"Decoder rebuilt: 512 -> {num_classes} classes (original architecture)")
    else:
        # Use custom architecture with penultimate layer: 512 -> penultimate_dim -> num_classes
        actual_model.decoder = nn.Sequential(OrderedDict([
            ("avgpool", old_avgpool),
            ("flatten", old_flatten),
            # Penultimate dense: 512 -> penultimate_dim (matches your Keras Dense(64, relu))
            ("penultimate_dense", nn.Linear(in_features, penultimate_dim)),
            ("penultimate_dense_relu", nn.ReLU(inplace=True)),
            ("penultimate_dense_drop", nn.Dropout(p=penultimate_dropout)),
            # Logits layer: penultimate_dim -> num_classes
            ("linear", nn.Linear(penultimate_dim, num_classes)),
            ("output", nn.Identity()),
        ]))
        print(f"Decoder rebuilt: 512 -> {penultimate_dim} (ReLU + Dropout) -> {num_classes} classes")
    
    # Optionally freeze backbone
    if freeze_backbone:
        print("Freezing backbone (V1-V4/IT), training decoder only")
        for name, param in actual_model.named_parameters():
            if not name.startswith("decoder."):
                param.requires_grad = False
    else:
        print("Training full model")
    
    # Return the unwrapped model (we'll handle device placement in training)
    return actual_model


def train_cornet(model, train_loader, val_loader=None, test_loader=None, epochs=10, lr=1e-4, 
                 device='cuda', weight_decay=1e-3, use_augmentation=True):
    """
    Train CORnet model with Validation and Testing steps.
    
    Enhanced with data augmentation and regularization to prevent overfitting.
    
    Parameters:
        model: CORnet model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (Optional, evaluated every epoch)
        test_loader: Test DataLoader (Optional, evaluated once at the end)
        epochs: number of training epochs
        lr: learning rate
        device: 'cuda' or 'cpu'
        weight_decay: L2 regularization strength (default: 1e-3)
        use_augmentation: whether to apply data augmentation during training (default: True)
        
    Returns:
        tuple: (trained_model, history)
    """
    import torch.optim as optim
    import torch.nn as nn
    import torchvision.transforms as transforms
    
    try:
        from tqdm.auto import tqdm
    except ImportError:
        from tqdm import tqdm
    
    model.to(device)
    
    # Add weight decay for strong regularization to prevent memorization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Define batch-level data augmentation
    # Applied on GPU for efficiency
    if use_augmentation:
        batch_augmenter = nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=10,          # Random rotation ±10 degrees
                translate=(0.1, 0.1), # Random translation ±10%
                scale=(0.9, 1.1)     # Random zoom 90-110%
            ),
            transforms.ColorJitter(
                brightness=0.2,      # Random brightness ±20%
                contrast=0.2,        # Random contrast ±20%
                saturation=0.1,      # Random saturation ±10%
                hue=0.05             # Random hue shift ±5%
            )
        ).to(device)
        print(f"Training with augmentation enabled (weight_decay={weight_decay})")
    else:
        batch_augmenter = None
        print(f"Training without augmentation (weight_decay={weight_decay})")
    
    # Initialize history with separate keys for train and val
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': []
    }
    
    print(f"Starting training for {epochs} epochs on {device}...")
    
    # Helper function for Evaluation
    def evaluate_pass(loader, description="Evaluating"):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = running_loss / len(loader)
        avg_acc = 100. * correct / total
        return avg_loss, avg_acc
    
    # Main Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training Pass
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Apply data augmentation during training (no gradient computation needed)
            if use_augmentation and batch_augmenter is not None:
                with torch.no_grad():
                    images = batch_augmenter(images)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': f'{running_loss/total:.4f}', 
                            'Acc': f'{100.*correct/total:.1f}%'})
        
        # Calculate Train Metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation Pass (If val_loader provided)
        val_str = ""
        if val_loader:
            val_loss, val_acc = evaluate_pass(val_loader, description="Validating")
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            val_str = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%{val_str}")
    
    # Final Test Pass (After all epochs)
    if test_loader:
        print("\nRunning Final Test Set Evaluation...")
        test_loss, test_acc = evaluate_pass(test_loader, description="Testing")
        print("="*60)
        print(f"FINAL TEST RESULT: Accuracy = {test_acc:.2f}% | Loss = {test_loss:.4f}")
        print("="*60)
        
        history['final_test_acc'] = test_acc
        history['final_test_loss'] = test_loss
    
    print("Training complete")
    return model, history