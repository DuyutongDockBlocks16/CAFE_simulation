from stable_baselines3.common.callbacks import BaseCallback
from config.training_config import SUCCESS_THRESHOLD

class SuccessCheckpointCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        
    def _on_step(self):
        if self.locals.get("dones", False) and self.locals["rewards"] > SUCCESS_THRESHOLD:
            self.model.save(f"{self.save_path}/success_model")
            if self.verbose:
                print(f"model has been saved, reward={self.locals['rewards']:.0f}")
        return True