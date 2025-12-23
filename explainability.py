import numpy as np
from mrs_prediction.model_zoo import init_model
import torch
import seaborn as sns
import matplotlib.pyplot as plt

model_checkpoint = "wandb/run-20251202_224951-w42p0gxo/files/SEResNext50MML.tar"
model = init_model("SEResNext50MML", out_channels=1)

checkpoint = torch.load(model_checkpoint, weights_only=True, map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])

weights = model.feedforward.weight
bias = model.feedforward.bias

W_THRESHOLD = 0.04

abs_numpy_weights = weights.abs().detach().numpy()

abs_numpy_weights[abs_numpy_weights < W_THRESHOLD] = 0.0

plt.figure(figsize=(20, 20), dpi=600)

sns.heatmap(abs_numpy_weights, vmin=0.0, vmax=np.max(abs_numpy_weights), cbar_kws={"label": f"Absolute Weight Value\n\nWeights < {W_THRESHOLD} are set to 0"})
# plt.tight_layout()
plt.xlim(1,1024)
plt.xticks([1, 512, 1024], ["$Weight_{1}$", "$Weight_{512}$", "$Weight_{1024}$"], rotation=0)
plt.ylim(1,512)
plt.yticks([1, 128, 256, 384, 512], ["$Neuron_{1}$", "$Neuron_{128}$", "$Neuron_{256}$", "$Neuron_{384}$", "$Neuron_{512}$"])
plt.xlabel("")
plt.ylabel("")
# plt.title("Weight Matrix of the Linear Layer Following Feature Concatenation (NCCT + Clinical)")
# plt.title("Weight Matrix of the Linear Layer Following Feature Concatenation (CTA + Clinical)")
# plt.title("Weight Matrix of the Linear Layer Following Feature Concatenation (NCCT + Clinical)")
plt.savefig("heatmap_ncct_clinical.png",bbox_inches = 'tight',
    pad_inches = 0)