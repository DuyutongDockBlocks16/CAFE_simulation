# CAFE: Collaborative Assembly Framework Environment

A multi-agent reinforcement learning framework for collaborative object manipulation and assembly tasks using MuJoCo physics simulation.

## 🎯 Project Overview

CAFE is a research framework that combines **Finite State Machines (FSM)** with **Reinforcement Learning (RL)** to enable efficient coordination between heterogeneous robotic agents in collaborative manipulation tasks. The system features hybrid control strategies where one robot follows deterministic FSM logic while another employs RL with action masking.

### Key Features

- 🤖 **Multi-Agent Coordination**: Two heterogeneous robots (6-DOF manipulator + mobile robot)
- 🔄 **Hybrid Control**: FSM-based deterministic control + RL with action masking
- ⚡ **MuJoCo Simulation**: High-fidelity physics simulation environment
- 📊 **Comprehensive Analytics**: Training progress tracking and visualization tools
- 🎯 **Task Flexibility**: Object picking, placing, and navigation coordination

## 🏗️ System Architecture

### Robot Configurations

#### Robot 1 (6-DOF Manipulator)
- **Control**: Deterministic Finite State Machine
- **Capabilities**: Precise manipulation, vacuum gripper
- **States**: IDLE → PICKING → MANIPULATION → PLACING → RESET

#### Robot 2 (Mobile Manipulator)  
- **Control**: Reinforcement Learning (PPO/MaskablePPO)
- **Capabilities**: Navigation, object picking/placing
- **Action Space**: 11 discrete actions with FSM-based masking
- **States**: 8 hierarchical FSM states constraining RL behavior

### Environment Variants

1. **V1**: Basic single-agent RL environment
2. **V2**: Multi-agent with simple coordination
3. **V3**: Advanced collaborative system with potential fields
4. **FSM Hybrid**: Specialized FSM-RL integration

## 📁 Project Structure

```
CAFE/
├── one_agent_training/           # Single agent RL training
│   ├── fsm_hybrid_robot_env.py   # Hybrid FSM-RL environment
│   ├── fsm_simplized_controller_training.py  # Training scripts
│   ├── simple_robot_env.py       # Basic single-agent environment
│   └── README.md                 # Single agent documentation
├── collab_training/              # Multi-agent collaborative training
│   ├── v3_collab_hybrid_robot_env.py        # V3 collaborative environment
│   ├── v3_collab_simplized_controller_training.py  # V3 training
│   ├── finite_state_controller.py           # FSM controller implementation
│   └── README.md                            # Collaborative training docs
├── data_analyze/                 # Analysis and visualization tools
│   ├── episode_reward_steps_analyze_thesis.py  # Episode analysis
│   ├── 9_curves.py              # Multi-run comparison plots
│   ├── avg_progress_timestep_per_object.py  # Performance metrics
│   └── other analysis scripts...
├── logs/                        # Training logs and experiment results
├── models/                      # Trained model checkpoints
├── venv/                       # Python virtual environment
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install gymnasium mujoco stable-baselines3 sb3-contrib
pip install numpy matplotlib pandas tikzplotlib
pip install torch torchvision  # For neural networks
```

### Training

#### Single Agent FSM-Hybrid Training
```bash
cd one_agent_training/
python fsm_simplized_controller_training.py
```

#### Collaborative V3 Training
```bash
cd collab_training/
python v3_collab_simplized_controller_training.py
```

### Testing Trained Models

```python
# Test FSM Hybrid model
from one_agent_training.fsm_simplized_controller_training import driver_model_test_single_episode
import gymnasium as gym

env = gym.make("FsmHybridMuJoCoEnv-v0")
driver_model_test_single_episode(env)
```

### Data Collection and Analysis

```bash
# Collect training data
cd collab_training/
python v3_collab_simplized_controller_training.py  # Run data collection

# Analyze results
cd data_analyze/
python episode_reward_steps_analyze_thesis.py
python 9_curves.py  # Compare multiple training runs
```

## 🎮 Environment Details

### Observation Space (61-dimensional)

**Current State Features (25D)**:
- Robot 2 position (x, y): `[2]`
- Robot 2 velocity (vx, vy): `[2]` 
- Robot 3 position (x, y): `[2]`
- Robot 3 velocity (vx, vy): `[2]`
- Target position (x, y): `[2]`
- Objects on placement areas: `[4]`
- Task status flags: `[5]`
- FSM state encoding: `[1]`
- Additional state info: `[5]`

**Historical Features (36D)**:
- Multi-scale temporal features (5, 10, 20, 50, 100, 200 steps)
- Position histories: `[12]`
- Velocity histories: `[12]`
- Target histories: `[12]`

### Action Space (11 discrete actions)

```python
ACTION_SPACE = {
    0: "Brake",               # Stop current motion
    1: "Keep Moving",         # Continue current action
    2: "Move to Picking",     # Navigate to object location
    3: "Pick",                # Pick up object
    4: "Move to Place 1",     # Navigate to placement area 1
    5: "Move to Place 2",     # Navigate to placement area 2
    6: "Place Upper",         # Place on upper level
    7: "Place Lower",         # Place on lower level
    8: "Return Origin",       # Return to starting position
    9: "Reverse",             # Move backward
    10: "Forward"             # Move forward
}
```

### FSM State Constraints (Action Masking)

Each FSM state restricts available actions:

```python
ALLOWED_ACTIONS = {
    RLRobotFiniteState.IDLE: [0, 1, 2],                    # Brake, Keep Moving, Move to Picking
    RLRobotFiniteState.NAVIGATE_TO_PICKING_POSITION: [0, 1, 9, 10],  # Navigation actions
    RLRobotFiniteState.PICKING_OBJECT: [3],                # Only picking allowed
    RLRobotFiniteState.MAKE_DECISION_ON_PLACING_POSITION: [4, 5],    # Choose placement area
    RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION_1: [0, 1, 9, 10], # Navigate to area 1
    RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION_2: [0, 1, 9, 10], # Navigate to area 2
    RLRobotFiniteState.PLACING_OBJECT_UPPER: [6],          # Only upper placement
    RLRobotFiniteState.PLACING_OBJECT_LOWER: [7],          # Only lower placement
    # ... other states
}
```

### Reward Function

**Multi-component reward system**:

1. **Task Completion Rewards**:
   - Object placement: `+100` per successful placement
   - Task completion: `+500` bonus for finishing episode

2. **Potential Field Rewards**:
   - Attractive potential (goals): Guides toward targets
   - Repulsive potential (obstacles): Collision avoidance
   - Dynamic reward based on distance changes

3. **Efficiency Penalties**:
   - Time penalty: `-0.1` per step to encourage efficiency
   - Collision penalty: `-10` for robot-robot collisions

4. **Constraint Violations**:
   - Placement capacity: `-5` for overfilled areas
   - Invalid actions: Prevented by action masking

## 📊 Training Configuration

### Hyperparameters

```python
# PPO Configuration
PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5
}

# Training Parameters
TRAINING_CONFIG = {
    "total_timesteps": 2_000_000,
    "log_interval": 1,
    "save_freq": 50_000,
    "eval_freq": 10_000,
    "eval_episodes": 5
}
```

### Model Architecture

```python
# Policy Network
policy_kwargs = {
    "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
    "activation_fn": torch.nn.ReLU
}
```

## 📈 Results and Analysis

### Performance Metrics

The system tracks multiple metrics:

- **Episode Reward**: Cumulative reward per episode
- **Episode Length**: Steps to completion
- **Success Rate**: Percentage of successful task completions
- **Collision Rate**: Robot-robot collision frequency
- **Efficiency**: Average timesteps per placed object

### Visualization Tools

1. **Episode Analysis**: `episode_reward_steps_analyze_thesis.py`
   - Reward curves with moving averages
   - Episode length distributions
   - Success rate trends

2. **Multi-Run Comparison**: `9_curves.py`
   - Compare different training runs
   - Outlier filtering and log-scale plotting
   - Convergence analysis

3. **Performance Metrics**: `avg_progress_timestep_per_object.py`
   - Policy progression analysis
   - Efficiency improvements over training rounds

## 🔧 Advanced Features

### Action Masking with MaskablePPO

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    return env.get_action_mask()

env = ActionMasker(env, mask_fn)
model = MaskablePPO("MlpPolicy", env, verbose=1)
```

### Potential Field Navigation

The system uses artificial potential fields for collision avoidance:

```python
# Attractive potential (toward goal)
attractive_potential = -0.5 * distance_to_target**2

# Repulsive potential (away from obstacles)
if distance_to_robot < influence_distance:
    repulsive_potential = 1.0 * (1/distance_to_robot - 1/influence_distance)**2

# Combined potential field reward
potential_reward = prev_potential - current_potential
```

### Multi-Scale Historical Features

The observation includes temporal context at multiple scales:

```python
HISTORY_SCALES = [5, 10, 20, 50, 100, 200]  # steps
# Captures short-term and long-term behavioral patterns
```

## 🐛 Troubleshooting

### Common Issues

1. **MuJoCo Installation**:
   ```bash
   pip install mujoco
   # Ensure MuJoCo binaries are properly installed
   ```

2. **Action Masking Errors**:
   ```python
   # Use correct parameter name
   action, _ = model.predict(obs, action_masks=action_mask)  # Note: 'masks' not 'mask'
   ```

3. **Environment Registration**:
   ```python
   import gymnasium as gym
   from one_agent_training.fsm_hybrid_robot_env import FsmHybridMuJoCoEnv
   gym.register(id="FsmHybridMuJoCoEnv-v0", entry_point=FsmHybridMuJoCoEnv)
   ```

### Performance Optimization

1. **Vectorized Environments**: Use `SubprocVecEnv` for parallel training
2. **Observation Normalization**: Apply `VecNormalize` wrapper
3. **Hyperparameter Tuning**: Use Optuna for automated optimization
