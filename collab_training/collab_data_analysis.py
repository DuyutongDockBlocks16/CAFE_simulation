import re
import csv

def parse_log_file(log_path, output_csv):
    # 匹配模式
    episode_pattern = re.compile(r"=== Data Collection Episode (\d+)/\d+ ===")
    step_pattern = re.compile(r"- Total MuJoCo steps executed: (\d+)")
    object_removed_pattern = re.compile(r"object(\d+):joint removed")
    robots_moved_apart_pattern = re.compile(r"🚨 Robots are too close! Moving them apart...")

    # 结局分类关键词
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

    # 按 episode 分割
    episodes = episode_pattern.split(content)
    # split会把分组里的episode号单独放出来，结构是 ["", "2", "episode2内容", "3", "episode3内容", ...]

    for i in range(1, len(episodes), 2):
        episode_id = episodes[i]
        episode_log = episodes[i+1]

        # 1. 结局
        outcome = "unknown"
        for key, pattern in outcomes.items():
            if re.search(pattern, episode_log):
                outcome = key
                break

        # 2. episode长度
        steps_match = step_pattern.search(episode_log)
        steps = int(steps_match.group(1)) if steps_match else 0

        # 3. 最大object进度
        object_ids = [int(m.group(1)) for m in object_removed_pattern.finditer(episode_log)]
        progress = max(object_ids) if object_ids else -1
        
        # 4. count the numbner of robot collision warnings
        robot_moved_apart_number = len(robots_moved_apart_pattern.findall(episode_log))
        if robot_moved_apart_number > 0:
            robot_moved_apart_number = robot_moved_apart_number
        else:
            robot_moved_apart_number = 0
        results.append([episode_id, outcome, steps, progress, robot_moved_apart_number])

    # 写入CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["episode", "outcome", "steps", "progress", "robot_moved_apart"])
        writer.writerows(results)

    print(f"解析完成，结果已保存到 {output_csv}")


if __name__ == "__main__":
    parse_log_file("data_collection_20251013_210155F1I.log", "data_collection_20251013_210155F1I.csv")
