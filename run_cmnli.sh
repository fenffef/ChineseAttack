#!/bin/bash
# 延时一小时
# sleep 1d


#!/bin/bash

# 定义模型和任务名称数组
models=("base" "smart" "creat" "r3f" "freelb")
tasks=("cmnli")

# 循环遍历不同的模型和任务
for model in "${models[@]}"
do
    for task in "${tasks[@]}"
    do
        model_name="fenffef/bert-$model-$task"
        dataset_name="fenffef/$task"
        
        echo "Running with model_name: $model_name, dataset_name: $dataset_name"
        python demo_$tasks.py --model_name "$model_name" --language "chinese" --dataset_name "$dataset_name" --split "validation[:500]"
        echo "Finished running with model_name: $model_name, dataset_name: $dataset_name"
    done
done

