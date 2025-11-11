## 🧭 Overview

**SpatialCL** is a *plug-and-play contrastive learning framework* designed for spatially structured modalities, including **RGB**, **thermal**, **RGB-D** data etc.
It robustly handles *intra-* and *inter-class variability*, enabling consistent embeddings across challenging datasets.

🧪 As a demonstration of its capabilities, **SpatialCL** has been applied in **[DiSCO 🔗](https://github.com/Olemou/SpatialCL)** — *Detection of Spills in Indoor environments using weakly supervised contrastive learning* — showcasing its practical impact in real-world spill detection scenarios.

⚙️ While the framework is **modality-agnostic** and can be extended to other dense spatial tasks, extending **SpatialCL** to sparse, graph-structured data such as **skeletons** represents an exciting direction for future work.

## Framework Architecture
  <p align="center">
  <img src="assets/framework.png" alt="SpatialCL Architecture" width="850"/>
</p>

**figure 1:** *Through the encoder, feature embeddings are extracted to obtain zij , which are subsequently normalized*. *From these embeddings, the cohesion*
*score cij is computed, representing how likely samples xi and xj remain compact within the same* *co-cluster. A binomial opinion is modeled as a Beta*
*probability density function (PDF), under the assumption of a bijective mapping between both representations. The uncertainty uij is then computed to*
*measure the confidence that xi and xj can be close within a cluster, enabling the model to avoid forcing samples of the same class together despite strong*
*visual differences. To ensure stable and gradual learning, a curriculum function  is introduced to guide progressive training and to*
*compute the adaptive weight wij , addressing intra-class variability. For inter-class modeling, the parameter β is computed as described in the schema above*,
*allowing the model to focus on hard negatives and enhance class separation. All these components are integrated into the final loss function Lij .*

## 🎯 Why DISCO Submodule of SpatialCL?
DISCO (Detection of Indoor Spills with Contrastive learning) addresses one of the most persistent challenges in computer vision: uncertainty under weak supervision. Traditional vision systems are typically designed and optimized for perception tasks involving rigid, well-structured objects with distinct geometric cues. However, these systems often fail when faced with visually ambiguous targets such as indoor liquid spills, whose irregular shapes, diffuse boundaries, and variable textures defy conventional object representations.

- ✨ *The difficulty arises from several intertwined factors:*
- ✨ *The absence of clear contours or well-defined shapes;*
- ✨ *Extreme intra-class variability in appearance and scale;*
- ✨ *Weak or inconsistent edge and texture cues;*
- ✨ *Frequent occlusion and foreground–background blending;*
- ✨ *A scarcity of reliable labeled examples; and*
- ✨ *environmental disturbances such as illumination changes, surface reflections, and sensor noise*.

## Key Features
- ✅ Handles **ambiguous and irregular objects** that standard vision models struggle with
- ✅ Supports: **RGB, thermal, depth, etc.**
- ✅ **Memory-optimized** contrastive learning for faster training
- ✅ Produces **highly discriminative embeddings** for downstream tasks
- ✅ Handles **class imbalance**
- ✅ Easy integration into existing PyTorch pipelines

## 📦Installation

# 🅐 Local Environment Setup

- ***1️⃣*** <b>Clone the repository</b>
<div align="left" style="margin-left:10%; height:30%;">
<pre>
<code>
git clone https://github.com/Olemou/SpatialCL.git
cd SpatialCL
</code>
</pre>
</div>
<!-- Setup Python virtual environment -->

- ***2️⃣*** <b>Setup a Python Virtual Environment (optional):</b>
<div align="left" style="margin-left:10%;">
<pre>
<code>

 ***1. Create virtual environment***
     python -m venv venv
***2. Activate the virtual environment***

***On Linux/macOS***
source venv/bin/activate

***On Windows (PowerShell)***
venv\Scripts\Activate.ps1

***On Windows (CMD)***
venv\Scripts\activate.bat
</code>
</pre>
</div>

 - ***3️⃣*** <b>Install dependencie:</b>
<div align="left" style="max-width:50%; margin-left:10%;">
<pre>
<code class="language-python">
pip install --upgrade pip
pip install -r requirements.txt
</code>
</pre>
</div>

 - ***4️⃣*** <b>Install local package in editable mode:</b>
<div align="left" style="max-width:50%; margin-left:10%;">
<pre>
<code class="language-python">
pip install -e .
</code>
</pre>
</div>

# 🅑 Usage of SpatialCL 

-  ### 1️⃣ PypI Installation ###
<div align="left" style="margin-left:10%; height:30%;">
<pre>
<code>
pip install -i https://test.pypi.org/simple/ spatialcl
</code>
</pre>
</div>
*After installing SpatialCL via ***pip***, you can leverage its comprehensive functionalities.*

- ### 2️⃣ Thermal Augmentation
Let's suppose the image is loaded and readable.

- <b>🧩 Occlusion </b>
<div align="left" style="max-width:50%; margin-left:10%;">
<pre>
<code class="language-python">
     from Spatialcl.thermal import occlusion
     aug_img = occlusion( img=image,mask_width_ratio=0.6,mask_height_ratio=0.2,max_attempts=5)
</code>
</pre>
</div>

- <b>🎛️ Contrast </b>
<div align="left" style="margin-left:10%;">
<pre>
<code class="language-python">
     from Spatialcl.thermal import contrast
     aug_img = contrast(img = image, alpha = 0.8)
</code>
</pre>
</div>

- <b>☀️ Mixed Brightness & Contrast<b>
<div align="left" style="margin-left:10%;">
<pre>
<code class="language-python">
     from Spatialcl.thermal import brightness_contrast
     aug_img = brightness_contrast(mg = image,brightness = 1, contrast = 0.6)
</code>
</pre>
</div>

- <b>🌀 Elastic Transfrormation </b>
<div align="left" style="margin-left:10%;">
<pre>
<code class="language-python">
     from Spatialcl.thermal import elastic
     aug_img = elastic(img = image,alpha = 1, sigma = 0.8)
</code>
</pre>
</div>

### 🚀 Compute Learning Mechanism

- ***🔸 uncertainty weight under weak supervision***
 *We computed uncertainty weights for  intra-class variability and class imbalance handling*
 <div align="left" style="margin-left:10%;">
<pre>
<code class="language-python">
from Spatialcl.uncertainty import co_cluster_uncertainty
z = torch.randn(4, 8)
img_id = torch.tensor([0, 1, 2, 3]) (img_id: augmented views of same image ids)
labels = torch.tensor([0, 1, 0, 1])
prior_weight = 2
uncertainy = co_cluster_uncertainty(z, labels, img_id)
</code>
</pre>
</div>

- ***🔸Curriculum learning mechanism***
 *The initially computed uncertainty weights are dynamically reweighted to guide the model through a progressive learning process, allowing it to first focus on easier examples and gradually incorporate harder, more ambiguous cases.*
 <div align="left" style="margin-left:10%;">
<pre>
<code class="language-python">
from Spatialcl.uncertainty import compute_weights_from_uncertainty
progressive_reweighting = compute_weights_from_uncertainty( 
     uncertainty=uncertainty_matrix, epoch=0 ,
     T = 100)
</code>
</pre>
</div>

- ***🔸Global Loss function in the learning process***
In this part, we compute the global learning mechanism in a single step, without intermediate stages. This approach simultaneously addresses challenges such as class imbalance, intra-class variability, and low inter-class similarity. It allows multiple modalities thermal, RGB, and RGB-D etc. to capture and reason about uncertainty effectively. By applying this mechanism to the spill dataset, which encompasses the full complexity of the data, the model is exposed to realistic challenges, highlighting the difficulties described earlier.
<div align="left" style="margin-left:10%;">
<pre>
<code class="language-python">
from Spatialcl.uncertainty import build_uwcl
img_id = torch.tensor([0, 1, 2, 1])
label = torch.tensor([0, 1, 0, 1])
z = torch.randn(4, 8)
output = build_uwcl(z=z, img_ids=img_id, labels=label, epoch=0, device="cpu")
</code>
</pre>
</div>

# 🅒 Training & Evalidation (DISCO)

### 🎯 Single-GPU (Non-Distributed) Training
***root: stands for your data folder path*** 
<div align="left" style="margin-left:10%;">
<pre>
<code class="language-python">
python train.py --batch_size 32 --num_workers 4 --root ./data --vit_variant base --temperatue 0.1 --num_epochs 50 ---dataset_class None --modality '{"rgb": False,"thermal": True}'
</code>
</pre>
</div>

### 🎯 Distributed Training One Node
***root: stands for your data folder path*** 
<div align="left" style="margin-left:10%;">
<pre>
<code class="language-python">
torchrun \
 --nnodes=1 \
  --nproc_per_node= 1 \
  --master_addr="127.0.0.1" \
  --master_port=29500 \
  train.py \
  --is_distributed \
  --batch_size 32
  --root ./data \ 
  --vit_variant base \ 
  --temperatue 0.1 \ 
  --num_epochs 50 \
  ---dataset_class None \
  --modality '{"rgb": False,"thermal": True}'
</code>
</pre>
</div>

# 🅓 Results (DISCO)
### 1️⃣ <b> Pretrained customize Vit-base </b>
<table>
  <tr>
    <!-- Column 1 -->
    <td>
      <h3 align="center">🧠 Pretrained Customized ViT-Base</h3>
    </td>
    <td style="text-align: center;">
      <a href="https://github.com/Olemou/SpatialCL/releases/tag/SpatialCl_(DISCO)" download>
        <button style="padding: 10px 20px; font-size: 16px; cursor: pointer;">
          Download Weights
        </button>
      </a>
    </td>
  </tr>
</table>

# 🅔 Visualization (DISCO)
### 1️⃣ <b> Original vs Attention-Map </b>
<table align="center">
  <tr>
    <td align="center"><b>Hot Water</b></td>
    <td align="center"><b>Last Block</b></td>
    <td align="center"><b>Attention Rollout</b></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/hot_water.png" alt="Hot Water" width="300"/></td>
    <td align="center"><img src="assets/hot_last_block.png" alt="Last Block" width="250"/></td>
    <td align="center"><img src="assets/hot_attention_rollout.png" alt="Attention Rollout" width="250"/></td>
  </tr>
</table>

### 2️⃣  <b> Downstream Task</b>
<table align="center">
  <tr>
    <td align="center"><b>Classification Metrics</b></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/disco.png" alt="Hot Water" width="800"/></td>
  </tr>
</table>

# 🅕 Beyond DISCO / SpatialCL applied to Anomaly Detection 
## key  features
- ✅ Handle  **class level variability** for downstream tasks
- ✅ Handles **class imbalance**
  


