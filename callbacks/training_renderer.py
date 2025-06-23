from stable_baselines3.common.callbacks import BaseCallback

class RenderCallback(BaseCallback):
    def __init__(self, env, render_freq=100):
        super().__init__()
        self.env = env
        self.render_freq = render_freq

    def _on_step(self):
        if self.n_calls % self.render_freq == 0:
            self.env.render()
        return True