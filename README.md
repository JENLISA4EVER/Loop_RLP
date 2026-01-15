#### Prepare
##### Train
conda create -n ouro python==3.10
pip install -r requirement.txt
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install deepspeed
pip install hydra-core --upgrade
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
##### Evaluate
cd lm-evaluation-harness
pip install -e .
cd evalplus
pip install -e .

#### 文件夹说明
- train/:
    - rlp_trainer.py:包含trainer的定义
- utils/:
    - dataset_pt.py:Dataset类和Collate类定义
    - entropy_cal.py:entropy的计算/logprob的计算
- config:包含评测和训练的设置
    - ouro_ntp_omnimath_eval: ntp-eval的config
    - ouro_reason_ntp_omnimath_eval: ntp-reason(generate)-eval的config
    - ouro_rlp_acc_omnimathntp: train-top16的config
    - ouro_rlp_acc_omnimath: train-online的config
- scripts/:训练脚本
    - a-train/:训练脚本文件夹
        - a-train_ntp.sh:top16训练
        - a-train_online.sh:train-online训练
    - a-evalntp/:ntp测试脚本文件夹
    - a-lm-evaluation:lm-evaluation-harness相关测试脚本
    - a-evalplus:evalplus相关测试脚本
- ouro/:ouro模型文件夹
    - modeling_ouro.py:修改后的model_ouro
- data_prepare/:数据切分文件夹
- evalplus/:eavlplus测试框架文件夹
- lm-evaluation-harness/:lm-evaluation-harness文件夹
- logs/:log输出文件夹
- results/:result输出文件夹
- train_ouro_rlp_acc_sx_v2.py:
    - 训练代码
- get_eval_sum.py:
    - 把evalntp results转为excel输出


#### Train执行
- 现在还是一机多卡
- 通过命令行参数/config_file(./configs/train/ouro_rlp_acc_omnimath.yaml)修改配置
- 数据在下载Omni_Math之后根据prepare_data/omni_math_split.py进行切分

```shell
bash ./scripts/a-train/a-train_online.sh
```

