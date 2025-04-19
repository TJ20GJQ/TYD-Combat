# 谷歌开源项目BERT源码解读（官方TF版）
BERT基于Transformer的编码器部分构建。Transformer是一种使用自注意力机制（Self-Attention）的神经网络架构，能够有效地捕捉序列中元素之间的依赖关系。

BERT的主要创新点：
- 预训练：BERT通过在大规模的文本语料库上进行无监督的预训练，学习到了丰富的语言知识和上下文表示。
- 微调：在预训练的基础上，通过在特定任务上进行微调，BERT能够在特定任务上取得较好的性能。

GLUE九大NLP任务：
MNLI：蕴含关系推断
QQP：问题对是否等价
QNLI：句子是都回答问句
SST-2：情感分析
CoLA：句子语言性判断
STS-B：语义相似
MRPC：句子对是都语义等价
RTE：蕴含关系推断
WNLI：蕴含关系推断

运行环境：
- Python 3.7.16
- tensorflow 1.14.0

运行准备：
- 下载[uncased_L-12_H-768_A-12模型](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip)到./GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12文件夹下

或下载[chinese_L-12_H-768_A-12模型](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)到./GLUE/BERT_BASE_DIR/chinese_L-12_H-768_A-12文件夹下
- 下载[GLUE数据集](https://gluebenchmark.com/tasks)到./GLUE/glue_data文件夹下

主要内容：
- run_classifier 基于BERT进行MRPC任务的训练/测试

运行训练命令：python run_classifier.py --task_name=MRPC --do_train=true --do_eval=true --data_dir=../GLUE/glue_data/MRPC --vocab_file=../GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=../GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=8 --learning_rate=2e-5 --num_train_epochs=1.0 --output_dir=../GLUE/output

运行测试命令：python run_classifier.py --task_name=MRPC --do_predict=true --data_dir=../GLUE/glue_data/MRPC --vocab_file=../GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=../GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --output_dir=../GLUE/output

详见[唐宇迪B站Pytorch课程](https://www.bilibili.com/video/BV1rg411d7KT?spm_id_from=333.788.videopod.episodes&vd_source=aaa85a47471179fcdb4e51e332c391e1&p=96)

项目来源：[@google-research/bert](https://github.com/google-research/bert)