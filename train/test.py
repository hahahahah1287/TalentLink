import torch
import os 
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================================
# 1. 评测配置与测试数据集
# ==========================================
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
lora_model_path = "/kaggle/working/router_lora_model"  # 刚才训练保存的目录

# 精选 15 条模型绝对没见过的测试数据
test_cases = os.open(test_router.json)

# ==========================================
# 2. 推理核心函数
# ==========================================
def predict(model, tokenizer, text):
    """
    统一的推理函数。
    提示词必须和 LLaMA-Factory 训练时(Alpaca格式)保持严格一致！
    """
    prompt = f"你是一个意图分类器。请将用户的输入分类为以下三种之一：JOB, CONTRACT, CHAT。\n用户输入：{text}\n分类结果："
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10, # 给 Base 模型稍微多一点 token，看看它会不会说废话
            temperature=0.01,  # 极低温度，消除随机性
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 截取模型新生成的部分
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # 清洗标点符号，方便严格比对
    result_clean = result.replace("。", "").replace(".", "").strip().upper()
    return result, result_clean

def run_evaluation(model, tokenizer, test_cases, model_name):
    print(f"\n{'='*50}")
    print(f"🚀 开始评测模型: {model_name}")
    print(f"{'='*50}")
    
    correct = 0
    start_time = time.time()
    
    for idx, case in enumerate(test_cases):
        raw_output, clean_output = predict(model, tokenizer, case["input"])
        
        # 严格比对：必须完全匹配 "JOB", "CONTRACT" 或 "CHAT"
        is_match = (clean_output == case["expected"])
        if is_match:
            correct += 1
            status = "✅ 命中"
        else:
            status = "❌ 错误"
            
        print(f"[{idx+1:02d}] {status} | 预期: {case['expected']:<8} | 实际原始输出: '{raw_output}'")

    end_time = time.time()
    accuracy = (correct / len(test_cases)) * 100
    avg_time = (end_time - start_time) / len(test_cases)
    
    print(f"\n📊 {model_name} 评测报告:")
    print(f"🎯 严格准确率: {accuracy:.1f}% ({correct}/{len(test_cases)})")
    print(f"⏱️ 平均推理耗时: {avg_time:.3f} 秒/条")
    return accuracy

# ==========================================
# 3. 执行测试：Base Model vs LoRA Model
# ==========================================

print("⏳ 正在加载 Tokenizer 和 基础模型 (Base Model)...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
# 使用 bfloat16 节省显存，device_map="auto" 会自动使用 T4 GPU
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
base_model.eval()

# 测试 1：跑未训练的基础模型
acc_base = run_evaluation(base_model, tokenizer, test_cases, "Qwen2.5-1.5B (未训练)")

print("\n⏳ 正在挂载 LoRA 权重 (融合进 Base Model)...")
# 测试 2：动态挂载 LoRA 并跑测试
# PeftModel 会用 LoRA 矩阵覆盖基础模型的注意力层
lora_model = PeftModel.from_pretrained(base_model, lora_model_path)
lora_model.eval()

acc_lora = run_evaluation(lora_model, tokenizer, test_cases, "Qwen2.5-1.5B + LoRA (微调后)")

# ==========================================
# 4. 总结
# ==========================================
print("\n" + "🏆"*20)
print(f"对比总结：")
print(f"未微调准确率: {acc_base:.1f}%")
print(f"微调后准确率: {acc_lora:.1f}%")
print(f"性能提升:     +{acc_lora - acc_base:.1f}%")
print("🏆"*20)