# [ACM MM 2024] Official implementation of the paper "Low-rank Prompt Interaction for Continual Vision-language Retrieval"


The code for image-text retrieval(retrieval) is based on  [S-prompts](https://github.com/iamwangyabin/S-Prompts).
The code for referring expression comprehension(grounding) is based on [GLIP](https://github.com/microsoft/GLIP). We follow [mdetr](https://github.com/ashkamath/mdetr) to prepare dataset.

# Image-text Retrieval
Annotations for image-text retrieval task is preprocessed based on the original annotations, which is available at the following Baidu Cloud link: https://pan.baidu.com/s/17wD4O8OwQSeV4qgBAIX4qQ?pwd=f0yv.

Run the following command for lpi. When the batch size is set to 64 and the prompt depth is set to 3. 

7159MB of GPU memory is required. 
For the retrieval task, only single GPU operation is supported.

```
# LPI
sh script/retrieval/lpi.sh
```

The results are generated in the retrieval/res directory, and post-processing of the results is performed using retrieval/res_handle/reshandle.py.

<!-- Some bugs need to be fixed... -->
# Referring Expression Comprehension
To run the grounding code, please first follow the instructions for installing the relevant packages for GLIP. First, install the necessary packages.
```
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo
pip install transformers 
```
Then, navigate to the lpi/grounding directory and install the maskrcnn_benchmark package in develop mode.
```
python setup.py build develop --user
```
Prepare GLIP-T(A) weight from this [GLIP-T(A) weight](https://huggingface.co/GLIPModel/GLIP/blob/main/glip_a_tiny_o365.pth) under `lpi/grounding/MODEL` folder. Prepare bert-base-uncased under `lpi/grounding/bert-base-uncased` folder, and prepare dataset [COCO2014](https://cocodataset.org/#download) under `lpi/grounding/DATASET` folder. We use annotations from [mdetr](https://github.com/ashkamath/mdetr/blob/main/.github/refexp.md).

Run the following command for website playground.
```
python grounding/webui/index.py
```
The playground's running results are shown as follows.
![webui](./assets/webui.png)

# Quickstart
To quickly illustrate our work, we present the core code below. Cross-modal and inter-task interactions are further enhanced based on this foundation using contrastive learning and interaction networks to achieve improved performance.
```
import torch
from torch import nn

class DecomposedPrompt(nn.Module):
    def __init__(self, layer_num, prompt_num, prompt_depth_vis, prompt_depth_text, r=4):
        super().__init__()
        self.d = r

        self.dim_1_share = nn.Parameter(torch.randn(layer_num, self.d))
        self.dim_2_visual = nn.Parameter(torch.randn(prompt_num,self.d))
        self.dim_2_textual = nn.Parameter(torch.randn(prompt_num,self.d))
        self.dim_3_visual = nn.Parameter(torch.rand(prompt_depth_vis, self.d))
        self.dim_3_textual = nn.Parameter(torch.rand(prompt_depth_text, self.d))

        nn.init.normal_(self.dim_1_share, std=0.5)
        nn.init.normal_(self.dim_2_visual, std=0.5)
        nn.init.normal_(self.dim_2_textual, std=0.5)
        nn.init.normal_(self.dim_3_visual, std=0.5)
        nn.init.normal_(self.dim_3_textual, std=0.5)
        # self.layerNorm = nn.LayerNorm(prompt_depth)
        self.scale = 1

    def forward(self):
        # d1
        dim_1_share = self.dim_1_share.view(-1,1,1,self.d)
        # d2
        dim_2_visual = self.dim_2_visual.view(1, -1, 1, self.d)
        dim_2_textual = self.dim_2_textual.view(1, -1, 1, self.d)
        # d3
        dim_3_visual = self.dim_3_visual.view(1, 1, -1, self.d)
        dim_3_textual = self.dim_3_textual.view(1, 1, -1, self.d)

        decomposed_prompt_visual = torch.mul(torch.mul(dim_1_share, dim_2_visual), dim_3_visual)
        decomposed_prompt_visual = torch.mean(decomposed_prompt_visual, dim=3)*self.scale

        decomposed_prompt_textual = torch.mul(torch.mul(dim_1_share, dim_2_textual), dim_3_textual)
        decomposed_prompt_textual = torch.mean(decomposed_prompt_textual, dim=3)*self.scale

        return decomposed_prompt_visual, decomposed_prompt_textual
```