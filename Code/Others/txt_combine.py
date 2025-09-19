import os
import re
import csv

def extract_cycles_strict(file_path):
    """
    严格根据 “Processing Cycle” 分组提取每轮的 20 个因子值。
    """
    factor_pattern = re.compile(r'Factor\s+([\w\d]+):\s+([-+]?[0-9]*\.?[0-9]+)')
    cycle_start_pattern = re.compile(r'Processing Cycle \d+/\d+')

    cycles = []
    current_cycle_values = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if cycle_start_pattern.search(line):
                # 如果已收集满20个值才认为是上一轮的cycle
                if len(current_cycle_values) == 20:
                    cycles.append(current_cycle_values)
                current_cycle_values = []
                continue

            match = factor_pattern.search(line)
            if match:
                value = float(match.group(2))
                current_cycle_values.append(value)

        # 文件结尾最后一轮处理
        if len(current_cycle_values) == 20:
            cycles.append(current_cycle_values)

    return cycles

def process_all_logs_to_csv(folder_path, output_csv_path):
    all_cycles = []
    factor_names = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            cycles = extract_cycles_strict(file_path)
            all_cycles.extend(cycles)

            # 获取因子顺序作为表头（取第一个文件的前 20 个即可）
            if not factor_names:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        match = re.match(r'.*Factor\s+([\w\d]+):', line)
                        if match:
                            factor_names.append(match.group(1))
                        if len(factor_names) == 20:
                            break

    # 写入CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ['cycle'] + factor_names
        writer.writerow(header)

        for i, values in enumerate(all_cycles, 1):
            writer.writerow([i] + values)

if __name__ == "__main__":
    folder_path = r'D:\lb\myCode\Landslide_susceptibility_mapping\R1_result\importance\processed_logs'
    output_csv = os.path.join(folder_path, "merged_cycles.csv")
    process_all_logs_to_csv(folder_path, output_csv)
    print(f"已合并所有日志，总计 {len(open(output_csv).readlines()) - 1} 个 cycle，结果保存为 CSV：{output_csv}")
