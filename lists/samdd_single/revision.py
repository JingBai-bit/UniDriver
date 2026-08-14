from pathlib import Path
import shutil

root = Path("../lists/samdd_single")

old_to_new = {
    0: 0,
    1: 1,
    2: 2,
    3: 2,
    4: 3,
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    9: 7,
}

target_files = []
for fold_dir in sorted(root.glob("fold_*")):
    for name in ["train.txt", "test.txt"]:
        file_path = fold_dir / name
        if file_path.exists():
            target_files.append(file_path)

if not target_files:
    raise FileNotFoundError(f"未找到需要处理的 train.txt 或 test.txt，检查目录是否存在：{root}")

for file_path in target_files:
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    shutil.copy2(file_path, backup_path)

    new_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{file_path} 第 {lineno} 行格式错误，应为 3 列，实际为 {len(parts)} 列：{line}")

            path, num_frames, old_label_str = parts

            try:
                old_label = int(old_label_str)
            except ValueError:
                raise ValueError(f"{file_path} 第 {lineno} 行标签不是整数：{old_label_str}")

            if old_label not in old_to_new:
                raise ValueError(f"{file_path} 第 {lineno} 行出现未定义旧标签：{old_label}")

            new_label = old_to_new[old_label]
            new_lines.append(f"{path} {num_frames} {new_label}\n")

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"已处理: {file_path}，备份: {backup_path}")