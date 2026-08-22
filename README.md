# 《音乐人工智能：从符号到声音》配套代码

## 目录结构

| 目录 | 对应章节 |
|---|---|
| `chapter01/` | 第 1 章　绪论与 Python 入门 |
| `chapter02/` | 第 2 章　符号表示 |
| `chapter03/` | 第 3 章　从符号到声音：MIDI渲染与信息鸿沟 |
| `chapter04/` | 第 4 章　符号理解与生成 |
| `chapter05/` | 第 5 章　音频表示 |
| `chapter06/` | 第 6 章　音频分类 |
| `chapter07/` | 第 7 章　音源分离 |
| `chapter08/` | 第 8 章　音频生成 |
| `chapter09/` | 第 9 章　音乐检索 |
| `chapter10/` | 第 10 章　评估哲学与基准测试 |

`datasets/` 与 `external/` 存放部分章节使用的数据集（GTZAN、FMA、CCMUSIC 等）和外部工具仓库（ACE-Step、YuE 等）。配套数据可通过此百度网盘链接获取：https://pan.baidu.com/s/1O8azgIaSgjz-0UNNB-GKVA?pwd=183j 提取码：183j；其它资源请自行下载。

## 运行方式

全书以 **Python 3.11.9** 为兼容性基线，各章依赖不同，**不要共用一个环境**。推荐每章单独建立虚拟环境：

```bash
python3.11 -m venv venv_chXX        # XX 为章号
source venv_chXX/bin/activate       # Windows: venv_chXX\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r chapterXX/requirements.txt
```

注意事项：

- 运行前请先执行各 Notebook 首单元的环境自检，确认解释器与依赖版本。
- 部分章节需要额外系统组件（如 FluidSynth + SoundFont、ffmpeg）或预训练模型权重，具体见正文或配套 Notebook 说明（第 1、6—10 章）。
- 第 8 章的重模型（MusicGen、AudioLDM2、Stable Audio、YuE、ACE-Step）需要各自独立的环境与硬件条件，无法在单一环境中全部运行。
