#!/bin/bash
# 延时一小时
# sleep 1d

task = "afqmc"

# Tnews
python demo_chinese.py --model_name "fenffef/bert-base-afqmc" --language "chinese" --dataset_name "fenffef/afqmc" --split "validation[:1000]"
# python demo_chinese.py --model_name "fenffef/bert-base-tnews" --language "chinese" --dataset_name "fenffef/tnews" --split "validation[5000:]"
python demo_chinese.py --model_name "fenffef/bert-creat-afqmc" --language "chinese" --dataset_name "fenffef/afqmc" --split "validation[:1000]"
# python demo_chinese.py --model_name "fenffef/bert-creat-tnews" --language "chinese" --dataset_name "fenffef/tnews" --split "validation[5000:]"
python demo_chinese.py --model_name "fenffef/bert-r3f-afqmc" --language "chinese" --dataset_name "fenffef/afqmc" --split "validation[:1000]"
# python demo_chinese.py --model_name "fenffef/bert-r3f-tnews" --language "chinese" --dataset_name "fenffef/tnews" --split "validation[5000:]"
python demo_chinese.py --model_name "fenffef/bert-freelb-afqmc" --language "chinese" --dataset_name "fenffef/afqmc" --split "validation[:1000]"
# python demo_chinese.py --model_name "fenffef/bert-freelb-tnews" --language "chinese" --dataset_name "fenffef/tnews" --split "validation[5000:]"
python demo_chinese.py --model_name "fenffef/bert-smart-afqmc" --language "chinese" --dataset_name "fenffef/afqmc" --split "validation[:1000]"
# python demo_chinese.py --model_name "fenffef/bert-smart-tnews" --language "chinese" --dataset_name "fenffef/tnews" --split "validation[5000:]"

