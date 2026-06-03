# Architecture choice: ResNet-18

We initially built the pipeline on a small from-scratch CNN (`SmallCNN`) to
validate the end-to-end flow — data loading, training, last-layer Laplace, OOD
scoring — quickly and cheaply. **From now on the main architecture is an
ImageNet-pretrained ResNet-18**, with `SmallCNN` kept only as a fast pipeline
smoke-test. The ResNet uses the stock torchvision BatchNorm; at inference it runs
in `eval()` mode, so BatchNorm uses fixed running statistics and predictions are
batch-independent.

## Why ResNet-18 specifically (not ResNet-50, ViT, DenseNet, …)

- **Mirrors how deep learning is actually used in practice.** The dominant
  real-world recipe — especially in medical imaging, where labelled data is
  scarce — is to take a pretrained backbone and fine-tune it, rather than train
  from scratch. An ImageNet-pretrained, fine-tuned ResNet is the textbook
  instance of that recipe, so our uncertainty/OOD findings speak to the setup
  practitioners actually deploy.
- **Standard reference architecture.** A baseline in essentially every
  medical-imaging-with-deep-learning paper since 2016.
- **Last-layer Laplace is well-defined for it.** Kristiadi et al. (2020), the
  last-layer-Bayesian illustration Li et al. cite, also uses ResNet-18, so our
  setup is methodologically aligned with the position paper's own example. The
  classifier head is named `fc`, exactly what our last-layer Laplace targets.
- **Matches the data scale.** MedMNIST images are tiny (28×28); a massive model
  like ViT-L would have its capacity wasted on so little information.
- **Comparable to the literature.** Li et al. (2025) and most OOD-detection work
  use ResNet-scale backbones, so our findings are directly comparable.

## ResNet-50 if time allows

A ResNet-50 variant is already implemented (`pretrained_resnet50`); it shares the
exact same code path, since the torchvision ResNets expose identical submodule
names (`conv1`, `layer1`…`layer4`, `fc`). It directly tests Li et al.'s "scaling
model and data size" intervention — does the conclusion about epistemic
uncertainty as an OOD detector change with model scale? We will run it **if
compute and time allow**; ResNet-18 stays the default because ResNet-50 roughly
doubles training time and memory for marginal gain on these small datasets.

## References

- Li et al. (2025), position paper on Bayesian deep learning for OOD detection.
- Kristiadi, Hein & Hennig (2020), *Being Bayesian, Even Just a Bit, Fixes
  Overconfidence in ReLU Networks* (last-layer Laplace on ResNet-18).
</content>
