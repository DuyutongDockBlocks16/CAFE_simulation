import numpy as np
from collections import deque

class FSMRewardBacktrack:
    def __init__(self, lookback_steps=50, success_bonus=1000, failure_penalty=-500):
        self.lookback_steps = lookback_steps
        self.success_bonus = success_bonus
        self.failure_penalty = failure_penalty
        self.episode_buffer = []
        
    def add_transition(self, state, action, reward, next_state, done, fsm_state):
        """添加转换到缓冲区"""
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'fsm_state': fsm_state,
            'step': len(self.episode_buffer)
        }
        self.episode_buffer.append(transition)
    
    def backtrack_rewards_on_success(self):
        """成功时回溯奖励给关键决策步骤"""
        if not self.episode_buffer:
            return
            
        # 🔥 识别关键FSM状态转换
        key_transitions = []
        for i, transition in enumerate(self.episode_buffer):
            if i > 0:
                prev_state = self.episode_buffer[i-1]['fsm_state']
                curr_state = transition['fsm_state']
                
                # 检测关键状态转换
                if self._is_critical_transition(prev_state, curr_state):
                    key_transitions.append(i-1)
        
        # 🔥 给关键转换分配额外奖励
        lookback_start = max(0, len(self.episode_buffer) - self.lookback_steps)
        for step_idx in key_transitions:
            if step_idx >= lookback_start:
                bonus = self.success_bonus * self._get_importance_weight(step_idx)
                self.episode_buffer[step_idx]['reward'] += bonus
                
    def backtrack_rewards_on_collision(self, collision_step):
        """碰撞时回溯惩罚给导致碰撞的行为"""
        if collision_step < self.lookback_steps:
            start_idx = 0
        else:
            start_idx = collision_step - self.lookback_steps
            
        # 🔥 分析导致碰撞的行为模式
        for i in range(start_idx, collision_step):
            transition = self.episode_buffer[i]
            
            # 惩罚过于激进的动作
            if self._is_aggressive_action(transition['action']):
                penalty_weight = self._get_collision_blame_weight(i, collision_step)
                self.episode_buffer[i]['reward'] += self.failure_penalty * penalty_weight
    
    def _is_critical_transition(self, prev_state, curr_state):
        """判断是否为关键状态转换"""
        from config.env_config import RLRobotFiniteState
        
        critical_transitions = [
            (RLRobotFiniteState.IDLE, RLRobotFiniteState.NAVIGATE_TO_PICKING_POSITION),
            (RLRobotFiniteState.PICKING_OBJECT, RLRobotFiniteState.MAKE_DECISION_ON_PLACING_POSITION),
            (RLRobotFiniteState.PLACING_OBJECT, RLRobotFiniteState.NAVIGATE_TO_PICKING_POSITION),
        ]
        
        return (prev_state, curr_state) in critical_transitions
    
    def _is_aggressive_action(self, action):
        """判断是否为激进动作"""
        # 检测可能导致碰撞的动作模式
        risky_actions = [9, 10]  # 前进后退调整动作
        return action in risky_actions
    
    def _get_importance_weight(self, step_idx):
        """计算步骤重要性权重（越靠近结束权重越高）"""
        total_steps = len(self.episode_buffer)
        return (step_idx + 1) / total_steps
    
    def _get_collision_blame_weight(self, action_step, collision_step):
        """计算碰撞责任权重（越接近碰撞时间权重越高）"""
        distance = collision_step - action_step
        return np.exp(-distance / 10.0)  # 指数衰减
    
    def get_modified_transitions(self):
        """获取修改后的转换"""
        return self.episode_buffer.copy()
    
    def reset(self):
        """重置缓冲区"""
        self.episode_buffer.clear()

# 在您的环境中集成
class FsmHybridMuJoCoEnvWithBacktrack(FsmHybridMuJoCoEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_backtrack = FSMRewardBacktrack()
        
    def step(self, action):
        obs, reward, terminated, truncated, info, action_switch = self._original_step(action)
        
        # 添加到回溯缓冲区
        self.reward_backtrack.add_transition(
            state=self._get_obs(),
            action=action,
            reward=reward,
            next_state=obs,
            done=terminated or truncated,
            fsm_state=self.second_robot_status
        )
        
        # 如果episode结束，进行奖励回溯
        if terminated or truncated:
            if terminated and reward > 5000:  # 成功完成
                self.reward_backtrack.backtrack_rewards_on_success()
            elif self._check_robot_robot_collision():  # 碰撞
                self.reward_backtrack.backtrack_rewards_on_collision(self.current_step)
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.reward_backtrack.reset()
        return super().reset(**kwargs)