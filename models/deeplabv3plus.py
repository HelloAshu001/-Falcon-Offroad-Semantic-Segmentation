import segmentation_models_pytorch as smp

def get_model(num_classes: int):
    """
    Switched to ResNet-34 to ensure 
    faster training while maintaining high accuracy.
    """
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",      
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    return model
