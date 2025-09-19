import re

def process_log_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_lines = []
    sum_values = []  # 存储每个 cycle 的累计总和
    current_cycle = 0
    current_cycle_raw = []  # 当前 cycle 所有平均值（用于计算）
    new_cycle_lines = []  # 当前 cycle 所有行，处理完再加入到结果中

    factor_line_pattern = re.compile(r'(.*Factor\s+[\w\d]+:\s+)([-+]?[0-9]*\.?[0-9]+)')

    for line in lines:
        # 发现新的 cycle
        cycle_match = re.search(r'Processing Cycle (\d+)/\d+', line)
        if cycle_match:
            # 如果上一轮已经读取了 20 个因子，则处理并保存
            if current_cycle_raw:
                actual_values = []
                if current_cycle == 1:
                    actual_values = current_cycle_raw
                else:
                    prev_sum = sum_values[-1]
                    for i in range(20):
                        curr_total = current_cycle_raw[i] * current_cycle
                        actual = curr_total - prev_sum[i]
                        actual_values.append(actual)
                # 写入新值
                for i in range(20):
                    prefix = re.match(r'(.*Factor\s+[\w\d]+:\s+)', new_cycle_lines[i]).group(1)
                    new_line = f"{prefix}{actual_values[i]:.6f}\n"
                    processed_lines.append(new_line)

                # 更新累计总和
                if current_cycle == 1:
                    sum_values.append(actual_values)
                else:
                    sum_values.append([s + a for s, a in zip(sum_values[-1], actual_values)])

                current_cycle_raw = []
                new_cycle_lines = []

            # 当前为新的 cycle
            current_cycle = int(cycle_match.group(1))
            processed_lines.append(line)
            continue

        value_match = factor_line_pattern.match(line)
        if value_match:
            value = float(value_match.group(2))
            current_cycle_raw.append(value)
            new_cycle_lines.append(line)
        else:
            # 非因子值的其他普通行，直接保留
            processed_lines.append(line)

    # 最后一轮 cycle 数据处理（结束前未触发 cycle 换行）
    if current_cycle_raw and len(current_cycle_raw) == 20:
        actual_values = []
        if current_cycle == 1:
            actual_values = current_cycle_raw
        else:
            prev_sum = sum_values[-1]
            for i in range(20):
                curr_total = current_cycle_raw[i] * current_cycle
                actual = curr_total - prev_sum[i]
                actual_values.append(actual)
        for i in range(20):
            prefix = re.match(r'(.*Factor\s+[\w\d]+:\s+)', new_cycle_lines[i]).group(1)
            new_line = f"{prefix}{actual_values[i]:.6f}\n"
            processed_lines.append(new_line)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)

if __name__ == "__main__":
    root_path = r'D:\lb\myCode\Landslide_susceptibility_mapping\R1_result\importance\3'
    input_log = f"{root_path}/importance_log.txt"
    output_log = f"{root_path}/importance_log_processed.txt"
    process_log_file(input_log, output_log)
    print(f"处理完成，结果已保存至 {output_log}")
