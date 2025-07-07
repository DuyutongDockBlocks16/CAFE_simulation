from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class LearningRateScheduler(BaseCallback):
    def __init__(self, 
                 initial_lr=1e-4, 
                 final_lr=1e-5, 
                 total_timesteps=2_400_000,
                 schedule_type='linear',
                 verbose=1):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_timesteps = total_timesteps
        self.schedule_type = schedule_type
        
    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        
        if self.schedule_type == 'linear':
            new_lr = self.initial_lr - (self.initial_lr - self.final_lr) * progress
        elif self.schedule_type == 'exponential':
            decay_rate = np.log(self.final_lr / self.initial_lr)
            new_lr = self.initial_lr * np.exp(decay_rate * progress)
        else:
            new_lr = self.initial_lr
        
        self.model.learning_rate = new_lr
        
        if self.num_timesteps % 50000 == 0 and self.verbose > 0:
            print(f"📚 Step {self.num_timesteps}: learning_rate = {new_lr:.2e}")
        
        return True