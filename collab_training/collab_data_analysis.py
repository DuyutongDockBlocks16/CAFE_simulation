import re
import csv

def parse_log_file(log_path, output_csv):
    # Matching patterns
    episode_pattern = re.compile(r"=== Data Collection Episode (\d+)/\d+ ===")
    step_pattern = re.compile(r"- Total MuJoCo steps executed: (\d+)")
    object_removed_pattern = re.compile(r"object(\d+):joint removed")
    robots_moved_apart_pattern = re.compile(r"🚨 Robots are too close! Moving them apart...")

    # Episode outcome classification keywords
    outcomes = {
        "timeout": r"Maximum steps reached, terminating episode.",
        "robot_collision": r"Robot-robot collision detected! Terminating episode.",
        "env_collision": r"Robot collision with forbidden area detected! Terminating episode.",
        "task_completed": r"Task completed successfully! Terminating episode.",
        "sim_error": r"(⚠️ QACC error detected, terminating episode.|Object-floor collision detected! Terminating episode.)",
        
    }

    results = []

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by episode
    episodes = episode_pattern.split(content)
    # split will put the episode number from the group separately, structure is ["", "2", "episode2_content", "3", "episode3_content", ...]

    for i in range(1, len(episodes), 2):
        episode_id = episodes[i]
        episode_log = episodes[i+1]

        # 1. Outcome
        outcome = "unknown"
        for key, pattern in outcomes.items():
            if re.search(pattern, episode_log):
                outcome = key
                break

        # 2. Episode length
        steps_match = step_pattern.search(episode_log)
        steps = int(steps_match.group(1)) if steps_match else 0

        # 3. Maximum object progress
        object_ids = [int(m.group(1)) for m in object_removed_pattern.finditer(episode_log)]
        progress = max(object_ids) if object_ids else -1
        
        # 4. Count the number of robot collision warnings
        robot_moved_apart_number = len(robots_moved_apart_pattern.findall(episode_log))
        if robot_moved_apart_number > 0:
            robot_moved_apart_number = robot_moved_apart_number
        else:
            robot_moved_apart_number = 0
        results.append([episode_id, outcome, steps, progress, robot_moved_apart_number])

    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["episode", "outcome", "steps", "progress", "robot_moved_apart"])
        writer.writerows(results)

    print(f"Parsing completed, results saved to {output_csv}")


if __name__ == "__main__":
    parse_log_file("data_collection_20251013_210155F1I.log", "data_collection_20251013_210155F1I.csv")
