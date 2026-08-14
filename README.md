# UniDriver: Multi-Task Vision-Language Adaptation for Generalizable Driver Distraction Detection

<p align="center">
    <img src="./img/framework.png" alt="Overview" width="800"/>
</p>

## Abstract

Driver distraction detection is critical for improving road safety. Recently,
CLIP-based vision-language models have demonstrated strong potential for this
task. However, existing methods still face challenges including limited task
adaptation, insufficient spatiotemporal modeling, and poor generalization across
drivers and scenarios.

To address these issues, we propose UniDriver, a multimodal multi-task adaptive
framework for video-based driver distraction detection. Specifically, we design
lightweight adapters in both the visual and textual branches to enable
parameter-efficient task adaptation. Within the visual adapter, we develop a
temporal enhancement mechanism to model global temporal relations and local
temporal differences. It produces compact spatiotemporal representations for
capturing dynamic distracted behaviors. Meanwhile, the textual adapter enhances
text representations to better capture driver distraction semantics.

Moreover, we present a multi-task joint learning strategy with complementary
supervisory signals, which encourages the model to learn stable and
discriminative representations in a unified vision-language semantic space and
improves generalization across drivers and scenarios. Experiments on three
public distracted driving datasets show that UniDriver outperforms
state-of-the-art methods in accuracy and generalization, validating its
effectiveness.

## Setup and Run

```bash
git clone https://github.com/JingBai-bit/UniDriver.git
cd UniDriver
pip install -r requirements.txt
cd scripts
sh train_driveract.sh
```

## Configuration

Before running experiments, edit `configs/config.py`:

- Set the pretrained vision-language checkpoint paths.
- Set the dataset roots for the datasets you want to use.
- Keep dataset split files under `lists/` or update the corresponding paths.

Large checkpoints, logs, and experiment outputs are not included in this
repository.
