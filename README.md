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

- ***PyPI***
<div align="left" style="margin-left:10%;">
<pre>
<code class="language-python">
pip install -i https://test.pypi.org/simple/ spatialcl==0.3.4
</code>
</pre>
</div>

- ***Clone the repository***
<div align="left" style="margin-left:10%;">
<pre>
<code class="language-python">
git clone https://github.com/Olemou/SpatialCL.git 
cd Spatialcl
</code>
</pre>
</div>

- ***Create and activate a virtual environment (optional):***
 <div align="left" style="margin-left:10%;">
<pre>
<code class="language-python">
 python -m venv venv
   ---On Linux/macOS---
    source venv/bin/activate
    --On Windows---
    venv\Scripts\activate
</code>
</pre>
</div>

 - ***Install dependencies:***
<div align="left" style="max-width:50%; margin-left:10%;">
<pre>
<code class="language-python">
 pip install --upgrade pip
pip install -r requirements.txt
</code>
</pre>
</div>

## 🎯 Usage of SpatialCL 
*After installing SpatialCL via ***pip***, you can leverage its comprehensive functionalities.*

### 🚀 Thermal Augmentation
Let's suppose the image is loaded and readable.

- ***🧩 Occlusion***
<div align="left" style="max-width:50%; margin-left:10%;">
<pre>
<code class="language-python">
from Spatialcl.thermal import thermal_occlusion
thermal_occlusion(
    img=image,
    mask_width_ratio=0.6,
    mask_height_ratio=0.2,
    max_attempts=5,
)
</code>
</pre>
</div>

- *** 🎛️ Contrast***
<div align="left" style="margin-left:10%;">
<pre>
<code class="language-python">
from Spatialcl.thermal import thermal_contrast
thermal_contrast(
   img = image, alpha = 0.8
)
</code>
</pre>
</div>

- *** ☀️ Mixed Brightness & Contrast***
<div align="left" style="margin-left:10%;">
<pre>
<code class="language-python">
from Spatialcl.thermal import brightness_contrast
brightness_contrast(
     img = image,
     brightness = 1
     contrast = 0.6,
)
</code>
</pre>
</div>

- ***🌀 Mixed Brightness & Contrast***
<div align="left" style="margin-left:10%;">
<pre>
<code class="language-python">
from Spatialcl.thermal import elastic_transform
elastic_transform(
     img = image,
     alpha = 1
     sigma = 0.8,
)
</code>
</pre>
</div>


