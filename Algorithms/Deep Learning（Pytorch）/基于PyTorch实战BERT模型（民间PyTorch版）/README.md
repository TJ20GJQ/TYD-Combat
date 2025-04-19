# 基于PyTorch实战BERT模型（民间PyTorch版）
BERT基于Transformer的编码器部分构建。Transformer是一种使用自注意力机制（Self-Attention）的神经网络架构，能够有效地捕捉序列中元素之间的依赖关系。

BERT的主要创新点：
- 预训练：BERT通过在大规模的文本语料库上进行无监督的预训练，学习到了丰富的语言知识和上下文表示。
- 微调：在预训练的基础上，通过在特定任务上进行微调，BERT能够在特定任务上取得较好的性能。

运行环境：
- Python 3.9.21
- CUDA 12.6
- torch 2.6.0+cu124
- scikit-learn 1.6.1

运行准备：
- 下载[BERT预训练文件](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz)到./bert_pretrain文件夹下

主要内容：
- run 基于BERT、BERT+CNN等网络进行新闻分类的训练和测试

运行命令：python .\run.py --model bert

详见[唐宇迪B站Pytorch课程](https://www.bilibili.com/video/BV1rg411d7KT?spm_id_from=333.788.videopod.episodes&vd_source=aaa85a47471179fcdb4e51e332c391e1&p=118)

项目来源：[@649453932/Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)