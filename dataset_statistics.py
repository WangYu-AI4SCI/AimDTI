# 假设数据文件名是 data.txt
# 数据格式：drugID targetID drug target relation

def analyze_dataset(file_path):
    print("分析数据集统计信息...")
    drugs = set()
    targets = set()
    rel_1_count = 0
    rel_0_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue  # 跳过空行
            parts = line.strip().split(maxsplit=4)  # 按空格分成5列
            if len(parts) != 5:
                continue  # 跳过异常行
            
            drug_id, target_id, drug, target, relation = parts
            drugs.add(drug_id)
            targets.add(target_id)
            
            if relation == "1":
                rel_1_count += 1
            elif relation == "0":
                rel_0_count += 1

    print(f"Drug 数量: {len(drugs)}")
    print(f"Target 数量: {len(targets)}")
    print(f"relation=1 的组合数量: {rel_1_count}")
    print(f"relation=0 的组合数量: {rel_0_count}")


# 运行
analyze_dataset("data/KIBA/KIBA.txt")
