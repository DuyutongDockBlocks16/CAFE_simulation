# Learning-Based Multi-Robot Collaboration for Intelligent and Adaptive Robotic Systems

An research project of Aalto University for researching of multi-robot collaboration, supervised by Professor Valeriy Vyatkin and Professor Antti Oulasvirta, advised by Dr Polina Ovsiannikova, and also supported by Dr Jifei Deng and Dr Shuai Ma. This project was initiated as part of the academic project “Conscious Agents as a Foundation for a Collaborative Factory of the Future” (CAFE).

## 🎯 Project Overview

This project is a master thesis work. The thesis of the project is in the `paper/` folder. And there are two case studies of the topic.

The 1st topic is an exploratory study: 
1. ####  We built a simulation environment to duplicate a manufacturing environment in MuJoCo, which is a pick-and-place scene consists of two pick tables and four place plates and 10 workpieces. Then, we modelled a system which consists of a robotic arm and a AGV, which is the robot system (Prototype) that used for this research work.
1. ####  Created a Primary Robot (R1) based on the Prototype, and programed R1 to execute pick-and-place task in the simulation environment. 
1. ####  Created a Secondary Robot (R2) also based on the Prototype. Trained the R2 to navigation in the simulation environment using AGV, R2 to execute pick action and place action using arm.
1. ####  Put both R1 and R2 to simulation environment, trained a decision-making model to drive R2 to avoid coliision with R1 and also execute the task.

The 2st topic is an collaboration study: 
1. ####  Removed R1, and added R3 which is another robot based on the Prototype.
1. ####  Trained a shared model that can be used for both R2 and R3 to execute the pick-and-place tasks in the simulation environment and avoid collisions to each other, the shared model can also be used for independently execute the pick-and-place tasks for R2 and R3.



### Technology Stack

- **Simulation and Robot**: MuJoCo, WL-MiroPRO-6R200-05MM
- **Reinforcement Learning**: Stablebaseline3, gymnasium, PPO & Maskable PPO, MLP

## 📁 Project Structure

```
CAFE/
├── xml/
│   ├── scene_mirobot.xml               # Sim env for 1st study
│   ├── mirobot_with_fork_lift_with_car_robot1.xml # R1 for 1st study
│   ├── mirobot_with_fork_lift_with_car_robot2.xml # R2 for both studies
│   ├── collab_mirobot.xml              # Sim env for 2st study
│   └── mirobot_with_fork_lift_with_car_robot3.xml # R3 for 2nd study
├── first_robot_controller/
│   └── mirobot_controller.py           # R1 control program
├── util_threads/
│   ├── object_placer.py               # Workpiece placer of the Sim env
│   ├── object_remover_step_counter.py # Workpiece remover of the env
│   └── object_remover.py              # Workpiece remover of the env
├── navigation_training/    # Navigation training for R2 in 1st study
│   ├── sec_robot_env.py           # Training env
│   └── second_robot_training.py   # Training enter point
├── picking_training/       # Pick training for R2 in 1st study
│   ├── picking_env.py             # Training env
│   └── picking_training.py        # Training enter point
├── placing_training/       # Place training for R2 in 1st study
│   ├── placing_env.py             # Training env
│   └── placing_training.py        # Training enter point
├── one_agent_training/   # Decision-making training for R2 in 1st study
│   ├── fsm_hybrid_robot_env.py   # Training env
│   ├── fsm_simplized_controller_training.py  # Training enter point
├── collab_training/  # Multi-robot training for 2nd study
│   ├── v3_collab_hybrid_robot_env.py        # Training env
│   ├── v3_collab_simplized_controller_training.py  # Enter point
│   └── collab_data_analysis.py # Data analysis of  Multi-robot training
├── data_analyze/                
│   └── episode_reward_steps_analyze.py  # Episode analysis
├── logs/                        # Training logs and experiment results
├── models/                      # Trained model checkpoints
└── README.md                   # This file
```

## 🚀 Quick Start

#### Install requirements
```bash
pip install -r requirements.txt
```


### Using a exist model
#### Navigation model for 1st case study
```bash
cd navigation_training
python second_robot_training.py
```

#### Pick model for 1st case study
```bash
cd picking_training
python picking_training.py
```

#### Place model for 1st case study
```bash
cd placing_training
python placing_training.py
```

#### desision-making model for 1st case study
```bash
cd one_agent_training
python fsm_simplized_controller_training.py
```

#### Shared collaboration model for 2nt case study
```bash
cd collab_training
python v3_collab_simplized_controller_training.py
```


### Training your own model

Each training script provides multiple training modes. You can uncomment the appropriate line based on your needs:

#### Navigation model for 1st case study
```bash
cd navigation_training
python second_robot_training.py
```

**Training modes in `second_robot_training.py`:**
```python
if __name__ == "__main__":
    approach_env = gym.make("SecondRobotMuJoCoEnv-v0")
    
    # 🎯 Choose ONE of the following training modes:
    
    # Mode 1: Single environment training from scratch
    approach_model_training(approach_env)
    
    # Mode 2: Single environment training with existing model
    # approach_model_training(approach_env, load_model_path=APPROACHING_MODEL_NAME)
    
    # Mode 3: Parallel training from scratch (faster)
    # approach_model_training_parallel()
    
    # Mode 4: Parallel training with existing model
    # approach_model_training_parallel(load_model_path=APPROACHING_MODEL_NAME)
    
    # 🧪 Testing mode (comment out when training)
    # approaching_model_implementation(approach_env)
```

**Usage Instructions:**
1. **For training**: Comment out `approaching_model_implementation(approach_env)` line
2. **For testing**: Uncomment `approaching_model_implementation(approach_env)` and comment out training lines
3. **Choose training mode**: Uncomment one of the four training modes above

#### Pick model for 1st case study
```bash
cd picking_training
python picking_training.py
```

**Training modes:**
```python
# Mode 1: Train from scratch
picking_model_training(picking_env)

# Mode 2: Continue training from existing model
# picking_model_training(picking_env, load_model_path=PICKING_MODEL_NAME)

# Mode 3: Parallel training (recommended for faster training)
# picking_model_training_parallel()

# Mode 4: Parallel training with existing model
# picking_model_training_parallel(load_model_path=PICKING_MODEL_NAME)

# Testing mode (disable during training)
# picking_model_implementation(picking_env)
```

#### Place model for 1st case study
```bash
cd placing_training
python placing_training.py
```

**Training modes:**
```python
# Single/Parallel training options (same pattern as above)
placing_model_training(placing_env)                                    # From scratch
# placing_model_training(placing_env, load_model_path=PLACING_MODEL_NAME)  # Continue training
# placing_model_training_parallel()                                      # Parallel from scratch
# placing_model_training_parallel(load_model_path=PLACING_MODEL_NAME)    # Parallel continue

# Testing mode
# placing_model_implementation(placing_env)
```

#### Decision-making model for 1st case study
```bash
cd one_agent_training
python fsm_simplized_controller_training.py
```

**Training modes:**
```python
# FSM-based hybrid training
fsm_model_training(fsm_env)                                    # From scratch
# fsm_model_training(fsm_env, load_model_path=FSM_MODEL_NAME)    # Continue training
# fsm_model_training_parallel()                                 # Parallel training

# Testing mode
# driver_model_test_single_episode(fsm_env)
```

#### Shared collaboration model for 2nd case study
```bash
cd collab_training
python v3_collab_simplized_controller_training.py
```

**Training modes:**
```python
# Collaborative multi-agent training
collab_model_training(collab_env)                                    # From scratch
# collab_model_training(collab_env, load_model_path=COLLAB_MODEL_NAME) # Continue training
# collab_model_training_parallel()                                    # Parallel training

# Testing mode
# collab_model_test_episode(collab_env)
```

### 🎯 Training Mode Selection Guide

| Mode | Description | When to Use | Performance |
|------|-------------|-------------|-------------|
| **Single Environment** | Uses one environment instance | Small models, debugging | Slower but stable |
| **Parallel Environment** | Uses multiple environment instances | Large-scale training | Faster training |
| **From Scratch** | Starts training with random weights | New experiments | Fresh start |
| **Continue Training** | Loads existing model and continues | Improving existing models | Resume progress |

### 🔧 Training Configuration

**Before Training:**
1. **Set training mode**: Uncomment the desired training line
2. **Disable testing**: Comment out `*_model_implementation()` functions
3. **Check model paths**: Ensure `APPROACHING_MODEL_NAME`, `PICKING_MODEL_NAME`, etc. point to correct files
4. **Configure hyperparameters**: Modify training parameters in each script if needed

**Example Training Session:**
```python
# For fast training from scratch
approach_model_training_parallel()

# For continuing existing training
# approach_model_training_parallel(load_model_path=APPROACHING_MODEL_NAME)
```

**After Training:**
1. **Enable testing**: Uncomment `*_model_implementation()` function
2. **Disable training**: Comment out all training lines
3. **Run evaluation**: Execute the script to test trained model
