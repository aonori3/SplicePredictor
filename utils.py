from tensorflow.keras.callbacks import LearningRateScheduler

def lr_scheduler(epoch: int, lr: float) -> float:
    """
    Learning rate scheduler function.
    
    Args:
        epoch (int): Current epoch number.
        lr (float): Current learning rate.
    
    Returns:
        float: Updated learning rate.
    """
    decay_rate = 2
    decay_epoch = 5
    
    return lr / decay_rate if epoch > decay_epoch else lr

def get_callbacks() -> list:
    """
    Get a list of callbacks for model training.
    
    Returns:
        list: A list containing the LearningRateScheduler callback.
    """
    return [LearningRateScheduler(lr_scheduler, verbose=1)]