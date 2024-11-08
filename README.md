# [ACM MM 2024] Official implementation of the paper "Low-rank Prompt Interaction for Continual Vision-language Retrieval"


The code for image-text retrieval(retrieval) is based on  [S-prompts](https://github.com/iamwangyabin/S-Prompts).
The code for referring expression comprehension(grounding) is based on [GLIP](https://github.com/microsoft/GLIP).

# Image-text Retrieval
Annotations for image-text retrieval task is preprocessed based on the original annotations, which is available at the following Baidu Cloud link: https://pan.baidu.com/s/17wD4O8OwQSeV4qgBAIX4qQ?pwd=f0yv.

Run the following command for lpi. When the batch size is set to 64 and the prompt depth is set to 3, 7159MB of GPU memory is required. 
For the retrieval task, only single GPU operation is supported.

```
# LPI
sh script/retrieval/lpi.sh
```
<!-- Some bugs need to be fixed... -->
