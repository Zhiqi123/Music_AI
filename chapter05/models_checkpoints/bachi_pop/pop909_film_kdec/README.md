# 权重下载说明：best_model.pt

BACHI 官方 Pop Model 预训练权重（POP909-CL 训练的 film_kdec 变体），约 125 MB。与本书其他第三方预训练权重的做法一致，
该文件不随代码包分发，请从官方渠道下载：

- 官方发布页：https://huggingface.co/datasets/Itsuki-music/BACHI_Chord_Recognition/tree/main/pop909_film_kdec
- 代码仓库（MIT 许可）：https://github.com/AndyWeasley2004/BACHI_Chord_Recognition

## 下载步骤

1. 打开官方发布页，下载 `best_model.pt`（国内网络可借助 hf-mirror.com 镜像访问
   Hugging Face）；
2. 将文件放入本目录，与 `config.yaml`、`vocab.pkl` 放在一起
   （这两个小文件同样来自官方发布页，已随包原样提供）。

## 完整性校验（可选）

```bash
shasum -a 256 best_model.pt
8a42fdb0e671253c6ef30c2f95bbf47cd4e17c5ae4c442bebcb48819cc41bd88
```

放置完成后，chapter04/06_learning_symbolic_analysis.ipynb 会自动在本目录找到并加载该权重。
运行该 Notebook 还需按其说明另行克隆 BACHI 源码仓库。
