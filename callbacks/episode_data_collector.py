import json
import os
import numpy as np
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeBatchCollector(BaseCallback):
    """
    Callback class for collecting episode data in batches and writing to file.
    Collects episode rewards, steps, and other metrics, then writes them in batches
    to improve performance compared to writing after each episode.
    """
    
    def __init__(self, output_file="episode_data.jsonl", batch_size=10, verbose=1):
        """
        Initialize the episode batch collector.
        
        Args:
            output_file (str): Path to output file for episode data
            batch_size (int): Number of episodes to collect before writing to file
            verbose (int): Verbosity level (0=quiet, 1=info, 2=debug)
        """
        super().__init__(verbose)
        
        self.output_file = output_file
        self.batch_size = batch_size
        
        # Data collection variables
        self.episode_buffer = []  # Batch write buffer
        self.current_episode_rewards = []  # Current episode reward tracking
        self.current_episode_steps = 0  # Current episode step count
        self.episode_count = 0  # Total episode counter
        self.total_steps = 0  # Total training steps
        
        # Performance statistics
        self.total_episodes_written = 0
        self.write_operations = 0
        
        # Initialize output file
        self._init_file()
        
        print(f"📊 EpisodeBatchCollector initialized:")
        print(f"   Output file: {self.output_file}")
        print(f"   Batch size: {batch_size} episodes")
    
    def _init_file(self):
        """Initialize the output file with metadata header."""
        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Create file header with metadata
        metadata = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "batch_size": self.batch_size,
            }
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False)
            f.write('\n')
        
        print(f"📝 Initialized file: {self.output_file}")
    
    def _flush_batch(self):
        """Write buffered episodes to file in batch."""
        if not self.episode_buffer:
            return
        
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                for episode_data in self.episode_buffer:
                    json.dump(episode_data, f, ensure_ascii=False)
                    f.write('\n')
                f.flush()  # Ensure data is written to disk
            
            self.total_episodes_written += len(self.episode_buffer)
            self.write_operations += 1
            
            if self.verbose >= 1:
                print(f"💾 Batch written: {len(self.episode_buffer)} episodes "
                      f"(Total: {self.total_episodes_written} episodes, "
                      f"Write ops: {self.write_operations})")
            
            self.episode_buffer.clear()
            
        except Exception as e:
            print(f"❌ Error writing batch: {e}")
    
    def _on_step(self):
        """Called at each training step to collect episode data."""
        # Get current step information
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        
        # Ensure list format (compatible with single and multi-environment)
        if not isinstance(rewards, (list, tuple, np.ndarray)):
            rewards = [rewards]
        if not isinstance(dones, (list, tuple, np.ndarray)):
            dones = [dones]
        
        # Initialize tracking lists if not already done
        if len(self.current_episode_rewards) == 0:
            self.current_episode_rewards = [[] for _ in range(len(rewards))]
            self.current_episode_steps = [0] * len(rewards)
        
        # Process each environment
        for env_idx in range(len(rewards)):
            # Accumulate rewards and step count
            self.current_episode_rewards[env_idx].append(float(rewards[env_idx]))
            self.current_episode_steps[env_idx] += 1
            self.total_steps += 1
            
            # Check if episode is finished
            if dones[env_idx]:
                # Calculate episode statistics
                episode_total_reward = sum(self.current_episode_rewards[env_idx])
                episode_total_steps = self.current_episode_steps[env_idx]
                self.episode_count += 1
                
                # Build episode data structure
                episode_info = {
                    'episode_number': self.episode_count,
                    'env_index': env_idx,
                    'total_reward': round(episode_total_reward, 2),
                    'total_steps': episode_total_steps,
                    'average_reward_per_step': round(episode_total_reward / episode_total_steps, 4) if episode_total_steps > 0 else 0,
                    'timestamp': datetime.now().isoformat(),
                    'training_step': self.n_calls,
                    # Optional: save last few step rewards (for debugging)
                    'last_5_rewards': [round(r, 2) for r in self.current_episode_rewards[env_idx][-5:]]
                }
                
                # Add to batch buffer
                self.episode_buffer.append(episode_info)
                
                # Display information
                if self.verbose >= 1:
                    print(f"📈 Episode {self.episode_count} completed (Env {env_idx}):")
                    print(f"   Reward: {episode_total_reward:.2f}")
                    print(f"   Steps: {episode_total_steps}")
                    print(f"   Buffer: {len(self.episode_buffer)}/{self.batch_size}")
                
                # Check if batch write is needed
                if len(self.episode_buffer) >= self.batch_size:
                    self._flush_batch()
                
                # Reset tracking for this environment
                self.current_episode_rewards[env_idx] = []
                self.current_episode_steps[env_idx] = 0
        
        return True
    
    def get_stats(self):
        """Get collection statistics."""
        return {
            'episodes_completed': self.episode_count,
            'episodes_written_to_file': self.total_episodes_written,
            'episodes_in_buffer': len(self.episode_buffer),
            'total_training_steps': self.total_steps,
            'write_operations': self.write_operations,
            'output_file': self.output_file
        }
    
    def _on_training_end(self):
        """Clean up operations when training ends."""
        # Write remaining buffered data
        if self.episode_buffer:
            print(f"📝 Writing final batch: {len(self.episode_buffer)} episodes")
            self._flush_batch()
        
        # Write final statistics to file
        final_stats = {
            "final_metadata": {
                "end_time": datetime.now().isoformat(),
                "total_episodes": self.episode_count,
                "total_training_steps": self.total_steps,
                "write_operations": self.write_operations,
                "batch_size": self.batch_size
            }
        }
        
        with open(self.output_file, 'a', encoding='utf-8') as f:
            json.dump(final_stats, f, ensure_ascii=False)
            f.write('\n')
        
        # Display final statistics
        print(f"\n📊 ============ Episode Data Collection Complete ============")
        print(f"📊 Total Episodes: {self.episode_count}")
        print(f"📊 Written to file: {self.total_episodes_written} episodes")
        print(f"📊 Write operations: {self.write_operations}")
        print(f"📊 Total training steps: {self.total_steps:,}")
        print(f"📝 Output file: {self.output_file}")
        
        # Display file size information
        if os.path.exists(self.output_file):
            file_size = os.path.getsize(self.output_file)
            print(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")


def read_episode_data(filename):
    """
    Read and analyze episode data from JSONL file.
    
    Args:
        filename (str): Path to episode data file
        
    Returns:
        dict: Dictionary containing episodes, metadata, and final metadata
    """
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found!")
        return None
    
    episodes = []
    metadata = None
    final_metadata = None
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                if 'metadata' in data:
                    metadata = data['metadata']
                elif 'final_metadata' in data:
                    final_metadata = data['final_metadata']
                elif 'episode_number' in data:
                    episodes.append(data)
                    
            except json.JSONDecodeError:
                continue
    
    print(f"📊 Loaded episode data from: {filename}")
    print(f"   Episodes: {len(episodes)}")
    if metadata:
        print(f"   Start time: {metadata.get('start_time', 'Unknown')}")
    if final_metadata:
        print(f"   End time: {final_metadata.get('end_time', 'Unknown')}")
        print(f"   Total training steps: {final_metadata.get('total_training_steps', 'Unknown'):,}")
    
    if episodes:
        rewards = [ep['total_reward'] for ep in episodes]
        steps = [ep['total_steps'] for ep in episodes]
        
        print(f"📊 Episode Statistics:")
        print(f"   Reward - Mean: {np.mean(rewards):.2f}, "
              f"Min: {np.min(rewards):.2f}, Max: {np.max(rewards):.2f}")
        print(f"   Steps - Mean: {np.mean(steps):.1f}, "
              f"Min: {np.min(steps)}, Max: {np.max(steps)}")
    
    return {
        'episodes': episodes,
        'metadata': metadata,
        'final_metadata': final_metadata
    }