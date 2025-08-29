import click
import os
import yaml
import itertools
import torch
import json
from datetime import datetime
import multiprocessing
import gc
import sys

# 确保可以从同级目录导入 pard_infer
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from pard_infer import PardDreamInfer

def worker(kwargs_combo, queue):
    """
    在独立进程中运行单个评估任务。
    """
    try:
        # 使用组合的参数初始化推断器
        infer = PardDreamInfer(**kwargs_combo)
        # 运行评估并获取结果
        result_metrics = infer.eval()
        
        # 将结果附加到原始参数组合中
        kwargs_combo['result'] = result_metrics
        kwargs_combo['device'] = torch.cuda.get_device_name(0)
        
        # 将最终的字典放入队列
        queue.put(kwargs_combo)
    except Exception as e:
        print(f"工作进程出错: {e}")
        import traceback
        traceback.print_exc()
        # 放入错误信息以便主进程知道
        queue.put({"error": str(e), **kwargs_combo})
    finally:
        # 彻底清理资源
        del infer
        torch.cuda.empty_cache()
        gc.collect()

@click.command()
@click.option('-c', '--config_path', default='config/eval_dream.yaml', help='评估配置文件的路径。')
def main(config_path):
    """
    主函数，用于加载配置、创建实验网格并运行评估。
    """
    print(f"正在从 {config_path} 加载配置...")
    with open(config_path, "r", encoding="utf-8") as f:
        eval_configs = yaml.safe_load(f)
    
    all_results = []
    # 'eval' 键下可以有多个实验组
    for experiment_group in eval_configs.get('eval', []):
        keys = list(experiment_group.keys())
        # 将所有参数值列表化，即使它只有一个值
        value_lists = [v if isinstance(v, list) else [v] for v in experiment_group.values()]
        
        # 为该实验组创建所有参数组合
        for combination in itertools.product(*value_lists):
            kwargs = dict(zip(keys, combination))
            print("\n" + "="*50)
            print(f"正在启动评估任务，参数: {kwargs}")
            
            q = multiprocessing.Queue()
            # 'spawn' 启动方法对于 CUDA 更安全
            p = multiprocessing.Process(target=worker, args=(kwargs, q))
            p.start()
            
            # 从队列中获取结果 (阻塞直到有结果)
            result = q.get()
            p.join() # 等待进程完全结束
            
            if 'error' in result:
                print(f"评估任务失败: {result['error']}")
            else:
                all_results.append(result)
                print(f"评估任务完成。临时结果: \n{json.dumps(result, indent=2)}")
            print("="*50 + "\n")
    
    # --- 保存所有结果 ---
    if not all_results:
        print("没有成功完成的评估任务，不生成结果文件。")
        return

    print("\n\n" + "#"*50)
    print("所有评估任务已完成。最终结果汇总:")
    for res in all_results:
        print(json.dumps(res, indent=2))
    
    save_base_path = 'datas/result'
    os.makedirs(save_base_path, exist_ok=True)
    
    save_path = os.path.join(save_base_path, f"dream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    print(f"\n正在将所有结果保存到: {save_path}")
    
    with open(save_path, "w", encoding="utf-8") as f:
        for item in all_results:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")
    print("结果保存完毕。")

if __name__ == '__main__':
    # 设置 'spawn' 启动方法，这对于 PyTorch 和 CUDA 是推荐的
    multiprocessing.set_start_method('spawn', force=True)
    main()
