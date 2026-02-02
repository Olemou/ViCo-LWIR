## 🧭 Overview

**ViCo-LWIR** is a *plug-and-play contrastive learning framework* designed for spatially structured modalities(**thermal**) data, etc.
It robustly handles *intra-* and *inter-class variability*, enabling consistent embeddings across challenging datasets.

🧪 As a demonstration of its capabilities, **ViCo-LWIR** has been applied in **[Vico-LWIR 🔗](https://github.com/Olemou/ViCo-LWIR)** — *Detection of Spills in Indoor environments using weakly supervised contrastive learning* — showcasing its practical impact in real-world spill detection scenarios.

⚙️ While the framework is **modality-agnostic** and can be extended to other dense spatial tasks, extending **SpatialCL** to sparse, graph-structured data such as **skeletons** represents an exciting direction for future work.

## Framework Architecture
  <p align="center">
  <img src="assets/framework/framework.png" alt="SpatialCL Architecture" width="850"/>
</p>

**figure 1:** *Through the encoder, feature embeddings are extracted to obtain zij , which are subsequently normalized*. *From these embeddings, the cohesion*
*score cij is computed, representing how likely samples xi and xj remain compact within the same* *co-cluster. A binomial opinion is modeled as a Beta*
*probability density function (PDF), under the assumption of a bijective mapping between both representations. The uncertainty uij is then computed to*
*measure the confidence that xi and xj can be close within a cluster, enabling the model to avoid forcing samples of the same class together despite strong*
*visual differences. To ensure stable and gradual learning, a curriculum function  is introduced to guide progressive training and to*
*compute the adaptive weight wij , addressing intra-class variability. For inter-class modeling, the parameter β is computed as described in the schema above*,
*allowing the model to focus on hard negatives and enhance class separation. All these components are integrated into the final loss function Lij .*

## 🎯 Why ViCo-LWIR?
ViCo-LWIR (ViCo-LWIR: Vision-Based detection of Indoor Liquid Spills Using
Long-Wave Infrared Imaging and a Weakly Supervised Contrastive) addresses one of the most persistent challenges in computer vision: uncertainty under weak supervision. Traditional vision systems are typically designed and optimized for perception tasks involving rigid, well-structured objects with distinct geometric cues. However, these systems often fail when faced with visually ambiguous targets such as indoor liquid spills, whose irregular shapes, diffuse boundaries, and variable textures defy conventional object representations.

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
## Demo
![Demo](assets/demo.gif)

## 📦Installation

# 🅐 Local Environment Setup

- ***1️⃣*** <b>Clone the repository</b>
<div align="left" style="margin-left:10%; height:30%;">
<pre>
<code>
git clone https://github.com/Olemou/Vico-LWIR.git
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
pip install -r requirement.txt
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

# 🅑 Usage of ViCo-LWIR 
### Learn More
- [Visit our guidance for details](https://olemou.github.io/Vico-Clr_Usage/)


# 🅒 Training & Evalidation (ViCo-LWIR)

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

# 🅓 Results (ViCo-LWIR)
### 1️⃣ <b> Pretrained customize Vit-base </b>
<table>
  <tr>
    <!-- Column 1 -->
    <td>
      <h3 align="center">🧠 Pretrained Customized ViT-Base</h3>
    </td>
    <td style="text-align: center;">
      <a href="https://github.com/Olemou/ViCo-LWIR/releases/tag/ViCo-LWIR" download>
        <button style="padding: 10px 20px; font-size: 16px; cursor: pointer;">
          Download Weights
        </button>
      </a>
    </td>
  </tr>
</table>

### 2️⃣  <b> Downstream Task</b>
<table align="center">
  <tr>
    <td align="center"><b>Classification Metrics</b></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/disco/disco.png" alt="Hot Water" width="800"/></td>
  </tr>
</table>


# 🅔 Visualization (ViCo-LWIR)
### 1️⃣ <b> Original vs Attention-Map </b>
<table align="center">
  <tr>
    <td align="center"><b>Hot Water</b></td>
    <td align="center"><b>Last Block</b></td>
    <td align="center"><b>Attention Rollout</b></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/disco/hot_water.png" alt="Hot Water" width="300"/></td>
    <td align="center"><img src="assets/disco/hot_last_block.png" alt="Last Block" width="250"/></td>
    <td align="center"><img src="assets/disco/hot_attention_rollout.png" alt="Attention Rollout" width="250"/></td>
  </tr>
</table>

# 🅕 Beyond ViCo-LWIR / SpatialCL(ViCo-LWIR transform in a python packaging) applied to Anomaly Detection 
## key  features
- ✅ Handle  **class level variability** for downstream tasks
- ✅ Handles **class imbalance**
- <h3>✅ Dataset</h3>
    <p>
      <a href="https://github.com/Olemou/ViCo-LWIR/releases" target="_blank" style="text-decoration:none;">
        <span style="
          display:inline-block;
          background-color:#0078D7;
          color:white;
          padding:10px 20px;
          border-radius:8px;
          font-size:16px;
          font-weight:bold;
          cursor:pointer;">
          ⬇️ Download MVTec AD Dataset
        </span>
      </a>
    </p>

  
  ### 1️⃣ <b>Anomaly detection in wood construction materials</b>
  - <b> Original Vs Attention-Map</b>
<table align="center">
  <tr>
    <th align="center">Original</th>
    <th align="center">Last Block Attn</th>
  </tr>

  <tr>
    <td align="center"><img src="assets/anomaly/wood/color.png" width="200"></td>
    <td align="center"><img src="assets/anomaly/wood/color_attn.png" width="200"></td>
  </tr>

  <tr>
    <td align="center"><img src="assets/anomaly/wood/combine.png" width="200"></td>
    <td align="center"><img src="assets/anomaly/wood/combine_attn.png" width="200"></td>
  </tr>

  <tr>
    <td align="center"><img src="assets/anomaly/wood/hole.png" width="200"></td>
    <td align="center"><img src="assets/anomaly/wood/hole_attn.png" width="200"></td>
  </tr>

  <tr>
    <td align="center"><img src="assets/anomaly/wood/scratch.png" width="200"></td>
    <td align="center"><img src="assets/anomaly/wood/scratch_attn.png" width="200"></td>
  </tr>

  <tr>
    <td align="center"><img src="assets/anomaly/wood/water.png" width="200"></td>
    <td align="center"><img src="assets/anomaly/wood/floor_attn.png" width="200"></td>
  </tr>
</table>

 ### 2️⃣ <b> Transistor (electronic)</b>
   - <b> Original Vs Attention-Map</b>
<table align="center">
  <tr>
    <th align="center">Original</th>
    <th align="center">Last Block Attn</th>
  </tr>

  <tr>
    <td align="center"><img src="assets/anomaly/transitor/original_damage.png" width="200"></td>
    <td align="center"><img src="assets/anomaly/transitor/attn_damage.png" width="200"></td>
  </tr>

  <tr>
    <td align="center"><img src="assets/anomaly/transitor/bend_original.png" width="200"></td>
    <td align="center"><img src="assets/anomaly/transitor/bend_attn.png" width="200"></td>
  </tr>

</table
  
# Vico-LWIR article
### Learn More
 [Visit](https://drive.google.com/file/d/1n1PWURCi0AniymTkhyVbjZZlY_de8INF/view?usp=sharing)
  


