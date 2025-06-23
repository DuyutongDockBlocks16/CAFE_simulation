import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

log_name = "episode_data_20250623_142748.jsonl"

def read_episode_data(filename):
    """
    Read episode data from JSONL file and extract episode information.
    
    Args:
        filename (str): Path to the JSONL file
        
    Returns:
        tuple: (episodes_data, metadata, final_metadata)
    """
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found!")
        return None, None, None
    
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
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {line[:50]}...")
                continue
    
    print(f"📊 Successfully loaded {len(episodes)} episodes from {filename}")
    return episodes, metadata, final_metadata

def plot_episode_analysis(episodes, metadata=None, final_metadata=None, save_path=None):
    """
    Plot episode analysis with total_reward and total_steps over episode numbers.
    
    Args:
        episodes (list): List of episode data dictionaries
        metadata (dict): Metadata from the file
        final_metadata (dict): Final metadata from the file
        save_path (str): Path to save the plot image
    """
    if not episodes:
        print("❌ No episode data to plot!")
        return
    
    # Extract data for plotting
    episode_numbers = [ep['episode_number'] for ep in episodes]
    total_rewards = [ep['total_reward'] for ep in episodes]
    total_steps = [ep['total_steps'] for ep in episodes]
    avg_rewards_per_step = [ep['average_reward_per_step'] for ep in episodes]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Episode Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Reward per Episode
    ax1.plot(episode_numbers, total_rewards, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Episode Number')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Total Reward per Episode')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(total_rewards), color='r', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(total_rewards):.1f}')
    ax1.legend()
    
    # Plot 2: Total Steps per Episode
    ax2.plot(episode_numbers, total_steps, 'g-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Episode Number')
    ax2.set_ylabel('Total Steps')
    ax2.set_title('Total Steps per Episode')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=np.mean(total_steps), color='r', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(total_steps):.1f}')
    ax2.legend()
    
    # Plot 3: Average Reward per Step
    ax3.plot(episode_numbers, avg_rewards_per_step, 'orange', linewidth=2, marker='^', markersize=4)
    ax3.set_xlabel('Episode Number')
    ax3.set_ylabel('Average Reward per Step')
    ax3.set_title('Average Reward per Step')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=np.mean(avg_rewards_per_step), color='r', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(avg_rewards_per_step):.3f}')
    ax3.legend()
    
    # Plot 4: Reward vs Steps Scatter
    scatter = ax4.scatter(total_steps, total_rewards, c=episode_numbers, cmap='viridis', alpha=0.7, s=50)
    ax4.set_xlabel('Total Steps')
    ax4.set_ylabel('Total Reward')
    ax4.set_title('Reward vs Steps (colored by episode)')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Episode Number')
    
    # Add statistics text
    stats_text = f"""
    Statistics Summary:
    Total Episodes: {len(episodes)}
    Reward - Min: {np.min(total_rewards):.1f}, Max: {np.max(total_rewards):.1f}
    Steps - Min: {np.min(total_steps)}, Max: {np.max(total_steps)}
    """
    
    if metadata:
        stats_text += f"\nTraining Start: {metadata.get('start_time', 'Unknown')[:19]}"
    if final_metadata:
        stats_text += f"\nTotal Training Steps: {final_metadata.get('total_training_steps', 'Unknown'):,}"
    
    # Add text box with statistics
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    
    plt.show()

def print_episode_summary(episodes, metadata=None, final_metadata=None):
    """Print a summary of episode data."""
    if not episodes:
        print("❌ No episode data to summarize!")
        return
    
    total_rewards = [ep['total_reward'] for ep in episodes]
    total_steps = [ep['total_steps'] for ep in episodes]
    avg_rewards_per_step = [ep['average_reward_per_step'] for ep in episodes]
    
    print(f"\n📊 ============ Episode Data Summary ============")
    print(f"📊 Total Episodes: {len(episodes)}")
    print(f"📊 Episode Range: {min(ep['episode_number'] for ep in episodes)} - {max(ep['episode_number'] for ep in episodes)}")
    
    print(f"\n🎯 Reward Statistics:")
    print(f"   Mean: {np.mean(total_rewards):.2f}")
    print(f"   Std:  {np.std(total_rewards):.2f}")
    print(f"   Min:  {np.min(total_rewards):.2f}")
    print(f"   Max:  {np.max(total_rewards):.2f}")
    
    print(f"\n📏 Steps Statistics:")
    print(f"   Mean: {np.mean(total_steps):.1f}")
    print(f"   Std:  {np.std(total_steps):.1f}")
    print(f"   Min:  {np.min(total_steps)}")
    print(f"   Max:  {np.max(total_steps)}")
    
    print(f"\n⚡ Average Reward per Step:")
    print(f"   Mean: {np.mean(avg_rewards_per_step):.4f}")
    print(f"   Std:  {np.std(avg_rewards_per_step):.4f}")
    print(f"   Min:  {np.min(avg_rewards_per_step):.4f}")
    print(f"   Max:  {np.max(avg_rewards_per_step):.4f}")
    
    if metadata:
        print(f"\n📅 Training Info:")
        print(f"   Start Time: {metadata.get('start_time', 'Unknown')}")
        print(f"   Batch Size: {metadata.get('batch_size', 'Unknown')}")
    
    if final_metadata:
        print(f"   End Time: {final_metadata.get('end_time', 'Unknown')}")
        print(f"   Total Training Steps: {final_metadata.get('total_training_steps', 'Unknown'):,}")
        print(f"   Write Operations: {final_metadata.get('write_operations', 'Unknown')}")
    
    # Find best and worst episodes
    best_reward_idx = np.argmax(total_rewards)
    worst_reward_idx = np.argmin(total_rewards)
    
    print(f"\n🏆 Best Episode (Reward): Episode {episodes[best_reward_idx]['episode_number']}")
    print(f"   Reward: {episodes[best_reward_idx]['total_reward']:.2f}")
    print(f"   Steps: {episodes[best_reward_idx]['total_steps']}")
    
    print(f"\n📉 Worst Episode (Reward): Episode {episodes[worst_reward_idx]['episode_number']}")
    print(f"   Reward: {episodes[worst_reward_idx]['total_reward']:.2f}")
    print(f"   Steps: {episodes[worst_reward_idx]['total_steps']}")

def main():
    
    """Main function to run the episode analysis."""
    # File path - modify this to your actual file path
    filename = f"../logs/{log_name}"
    
    # Alternative: Ask user for filename
    # filename = input("Enter the path to your episode data file: ").strip()
    
    print(f"📂 Reading episode data from: {filename}")
    
    # Read data
    episodes, metadata, final_metadata = read_episode_data(filename)
    
    if not episodes:
        print("❌ No episode data found. Exiting.")
        return
    
    # Print summary
    print_episode_summary(episodes, metadata, final_metadata)
    
    # Create plot
    save_path = filename.replace('.jsonl', '_analysis.png')
    plot_episode_analysis(episodes, metadata, final_metadata, save_path)
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()