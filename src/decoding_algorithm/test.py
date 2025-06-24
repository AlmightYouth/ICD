import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from contrastive_decoding import ContrastiveDecoding

def main():
    parser = argparse.ArgumentParser(description="测试对比解码+前缀方法")
    
    parser.add_argument("--model_name", type=str, default="/root/autodl-tmp/ICD/model/llama-7b-chat-hf", 
                       help="基础模型路径（教师模型）")
    parser.add_argument("--amateur_model_name", type=str, default="/root/autodl-tmp/ICD/model/llama-7b-chat-hf", 
                       help="业余模型路径（学生模型）")
    parser.add_argument("--max_gpu_memory", type=int, default=39, help="最大GPU内存使用量(GB)")
    parser.add_argument("--num_gpus", type=int, default=1, help="用于基础模型的GPU数量")
    parser.add_argument("--amateur_model_nums_gpus", type=int, default=1, 
                       help="用于业余模型的GPU数量")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], 
                       help="使用设备")
    
    parser.add_argument("--input_text", type=str, 
                       default="Q: 请详细解释一下量子计算的基本原理", 
                       help="输入提示文本")
    parser.add_argument("--max_new_tokens", type=int, default=300, 
                       help="生成的最大新token数量")
    
    parser.add_argument("--first_gen_mode", type=str, default="baseline", 
                       choices=["baseline", "contrastive"], 
                       help="首轮回复生成模式")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p采样参数")
    parser.add_argument("--top_k", type=int, default=0, help="top-k采样参数")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度参数")
    parser.add_argument("--relative_top", type=float, default=0.1, 
                       help="相对顶部过滤参数")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--test_both", action="store_true", 
                       help="连续测试单轮和对比+前缀方法，并显示对比结果")
    
    args = parser.parse_args()

    print(f"初始化对比解码模型...")
    cd = ContrastiveDecoding(
        model_name=args.model_name,
        amateur_model_name=args.amateur_model_name,
        device=args.device,
        max_gpu_memory=args.max_gpu_memory,
        num_gpus=args.num_gpus,
        amateur_model_nums_gpus=args.amateur_model_nums_gpus
    )

    # 添加停止词（可选）
    cd.set_stop_words(["Human:", "User:", "Q:"])
    
    input_text = args.input_text
    print(f"输入提示: {input_text}")
    print("-" * 50)
    
    # 测试流程
    print("2. 使用对比解码模式生成回复:")
    contrastive_output,text_output = cd.generate(
        input_text=input_text,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        mode="dual-contrastive-decoding",
        verbose=True,
        remove_stop_words=True,
        relative_top=args.relative_top,
        first_gen_mode="greedy"  # 第一轮使用贪婪搜索
    )
    

    print(text_output)


if __name__ == "__main__":
    main()
