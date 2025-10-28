from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class EntCoefficientScheduler(BaseCallback):
    def __init__(self, 
                 initial_ent_coef=0.02, 
                 final_ent_coef=0.001, 
                 total_timesteps=2_400_000,
                 schedule_type='linear',  # 'linear', 'exponential', 'cosine'
                 verbose=1):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.total_timesteps = total_timesteps
        self.schedule_type = schedule_type
        self.training_start_timesteps = 0
        self.initialized = False
    
    def _on_training_start(self) -> None:
        self.training_start_timesteps = self.num_timesteps
        self.initialized = True

        print(f"🎯 EntScheduler starting...")
        print(f"   Model ent_coef before: {self.model.ent_coef:.6f}")
        self.model.ent_coef = self.initial_ent_coef
        print(f"   Model ent_coef after: {self.model.ent_coef:.6f}")
        print(f"   Will decay to: {self.final_ent_coef:.6f} over {self.total_timesteps:,} steps")
        
        
    def _on_step(self) -> bool:
        if not self.initialized:
            return True

        steps_since_start = self.num_timesteps - self.training_start_timesteps
        progress = min(steps_since_start / self.total_timesteps, 1.0)

        if self.schedule_type == 'linear':
            new_ent_coef = self.initial_ent_coef - (self.initial_ent_coef - self.final_ent_coef) * progress
        
        elif self.schedule_type == 'exponential':

            decay_rate = np.log(self.final_ent_coef / self.initial_ent_coef)
            new_ent_coef = self.initial_ent_coef * np.exp(decay_rate * progress)
        
        elif self.schedule_type == 'cosine':

            new_ent_coef = self.final_ent_coef + (self.initial_ent_coef - self.final_ent_coef) * \
                          (1 + np.cos(np.pi * progress)) / 2
        
        else:
            new_ent_coef = self.initial_ent_coef
        
        self.model.ent_coef = max(new_ent_coef, 0.0001)  # Ensure it doesn't become 0
        
        # More detailed logging
        if steps_since_start % 10000 == 0 and self.verbose > 0:
            print(f"📉 Total steps: {self.num_timesteps:,} | "
                  f"New steps: {steps_since_start:,} | "
                  f"Progress: {progress:.1%} | "
                  f"ent_coef: {self.model.ent_coef:.6f}")

        return True