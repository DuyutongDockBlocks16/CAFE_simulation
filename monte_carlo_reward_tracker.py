def __init__(self, action_repeat=40):
    # ... 现有代码 ...
    
    # 🔥 延迟奖励追踪系统
    self.pending_rewards = []  # 存储待确认的延迟奖励
    self.decision_events = []  # 存储关键决策事件
    
class DelayedRewardEvent:
    def __init__(self, event_type, step, action, context, potential_reward, timeout_steps=500):
        self.event_type = event_type  # "tray_choice", "path_choice", "timing_choice"
        self.step = step              # 决策发生的步数
        self.action = action          # 执行的动作
        self.context = context        # 决策时的环境状态
        self.potential_reward = potential_reward  # 潜在奖励值
        self.timeout_steps = timeout_steps  # 超时步数
        self.resolved = False         # 是否已解决
        self.final_reward = 0         # 最终确定的奖励

def _record_decision_event(self, event_type, action, potential_reward, timeout_steps=500):
    """记录关键决策事件"""
    context = {
        'robot1_pos': self.data.xpos[self.robot_1_rover_id][:2].copy(),
        'robot2_pos': self.data.xpos[self.robot_2_rover_id][:2].copy(),
        'robot2_status': self.second_robot_status,
        'tray_states': self._get_object_number_on_each_placing_place(),
        'robot1_carrying': self.first_robot_is_carrying,
    }
    
    event = DelayedRewardEvent(
        event_type=event_type,
        step=self.current_step,
        action=action,
        context=context,
        potential_reward=potential_reward,
        timeout_steps=timeout_steps
    )
    
    self.decision_events.append(event)
    print(f"🎯 记录延迟奖励事件: {event_type}, 步数={self.current_step}, 潜在奖励={potential_reward}")

def _check_and_resolve_delayed_rewards(self):
    """检查并解决延迟奖励"""
    total_delayed_reward = 0
    
    for event in self.decision_events:
        if event.resolved:
            continue
            
        # 检查是否超时
        if self.current_step - event.step > event.timeout_steps:
            event.resolved = True
            event.final_reward = event.potential_reward * 0.1  # 超时给予小额奖励
            total_delayed_reward += event.final_reward
            print(f"⏰ 延迟奖励超时: {event.event_type}, 最终奖励={event.final_reward}")
            continue
        
        # 🔥 根据事件类型检查是否可以解决
        if event.event_type == "tray_choice":
            resolved_reward = self._resolve_tray_choice_reward(event)
            if resolved_reward is not None:
                event.resolved = True
                event.final_reward = resolved_reward
                total_delayed_reward += resolved_reward
                print(f"✅ 托盘选择奖励确认: 最终奖励={resolved_reward}")
                
        elif event.event_type == "path_choice":
            resolved_reward = self._resolve_path_choice_reward(event)
            if resolved_reward is not None:
                event.resolved = True
                event.final_reward = resolved_reward
                total_delayed_reward += resolved_reward
                print(f"🛤️ 路径选择奖励确认: 最终奖励={resolved_reward}")
                
        elif event.event_type == "timing_choice":
            resolved_reward = self._resolve_timing_choice_reward(event)
            if resolved_reward is not None:
                event.resolved = True
                event.final_reward = resolved_reward
                total_delayed_reward += resolved_reward
                print(f"⏱️ 时机选择奖励确认: 最终奖励={resolved_reward}")
    
    # 🔥 清理已解决的事件
    self.decision_events = [e for e in self.decision_events if not e.resolved]
    
    return total_delayed_reward

def _resolve_tray_choice_reward(self, event):
    """解决托盘选择的延迟奖励"""
    # 只有当物体成功放置或失败时才能确定奖励
    if self.second_robot_status in [
        RLRobotFiniteState.NAVIGATE_TO_PICKING_POSITION,  # 成功完成放置，回到拾取
        RLRobotFiniteState.IDLE  # 或者回到空闲状态
    ] and self.second_robot_is_picking == False:
        
        # 检查放置是否成功
        current_tray_states = self._get_object_number_on_each_placing_place()
        previous_tray_states = event.context['tray_states']
        
        # 判断选择的托盘是否成功放置了物体
        if event.action == 4:  # 选择托盘1
            tray1_change = (current_tray_states[0] + current_tray_states[2]) - \
                          (previous_tray_states[0] + previous_tray_states[2])
            if tray1_change > 0:  # 成功放置
                # 进一步检查是否造成超载
                if current_tray_states[0] > 1 or current_tray_states[2] > 1:
                    return event.potential_reward * -0.5  # 放置成功但造成超载
                else:
                    return event.potential_reward * 1.0   # 完美放置
            else:
                return event.potential_reward * -0.8  # 选择了但没能放置
                
        elif event.action == 5:  # 选择托盘2
            tray2_change = (current_tray_states[1] + current_tray_states[3]) - \
                          (previous_tray_states[1] + previous_tray_states[3])
            if tray2_change > 0:
                if current_tray_states[1] > 1 or current_tray_states[3] > 1:
                    return event.potential_reward * -0.5
                else:
                    return event.potential_reward * 1.0
            else:
                return event.potential_reward * -0.8
    
    return None  # 还未到解决时机

def _resolve_path_choice_reward(self, event):
    """解决路径选择的延迟奖励"""
    # 检查是否已经到达目标或发生碰撞
    if hasattr(self, 'collision_occurred') and self.collision_occurred:
        return event.potential_reward * -1.0  # 路径选择导致碰撞
    
    # 检查是否成功到达目标
    if self.robot_2_target_position_x_y is not None:
        robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]
        target_pos = np.array(self.robot_2_target_position_x_y)
        distance = np.linalg.norm(target_pos - robot2_pos)
        
        if distance < 0.3:  # 成功到达目标
            # 计算效率奖励
            steps_taken = self.current_step - event.step
            efficiency_factor = max(0.1, 1.0 - steps_taken / 200.0)  # 步数越少效率越高
            return event.potential_reward * efficiency_factor
    
    return None

def _resolve_timing_choice_reward(self, event):
    """解决时机选择的延迟奖励"""
    # 检查等待/行动的时机是否合适
    current_robot1_pos = self.data.xpos[self.robot_1_rover_id][:2]
    decision_robot1_pos = event.context['robot1_pos']
    
    robot1_moved_distance = np.linalg.norm(current_robot1_pos - decision_robot1_pos)
    
    if event.action == 0:  # 选择等待
        # 如果robot1确实在移动，等待是好决策
        if robot1_moved_distance > 1.0:
            return event.potential_reward * 1.0  # 好的等待时机
        else:
            return event.potential_reward * -0.3  # 不必要的等待
    else:  # 选择行动
        # 如果robot1没有太大移动，行动是好决策
        if robot1_moved_distance < 0.5:
            return event.potential_reward * 1.0  # 好的行动时机
        else:
            return event.potential_reward * -0.5  # 可能的冲突
    
    return None