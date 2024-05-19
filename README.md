# CAN
The implementation of Novel Category Discovery Across Domains with Contrastive Learning and Adaptive Classifier (IJCNN2023)

Environment：

```
pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 
```
Run example:
```
wandb sweep sweep/opda/office_opda.yaml

# output:
# wandb: Creating sweep from: sweep/opda/office_opda.yaml
# wandb: Creating sweep with ID: eimkqhiu
# wandb: View sweep at: https://wandb.ai/lrh352/CAN/sweeps/eimkqhiu
# wandb: Run sweep agent with: wandb agent lrh352/CAN/eimkqhiu

wandb agent lrh352/CAN/eimkqhiu
```
