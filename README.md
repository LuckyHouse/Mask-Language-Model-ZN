# Mask-Language-Model-ZN
中文 mask language 模型
工程修改于https://github.com/huanghonggit/Mask-Language-Model

##快速上手

* 运行preprocess.py 处理分词后的数据，过滤掉过长的数据(hparams.enc_maxlen)
* 运行vocab.py 生成词表
* 运行 __main__.py
    * train() 首次训练
    * continue_train() 继续训练
    * test() 输出测试集Mask的TOPN概率 和 label的概率