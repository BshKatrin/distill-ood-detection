# Vocabulary

- ID (In-Distribution): Data drawn from the same distribution as the training data, or from the distribution the model is expected to encounter during deployment.

- OOD (Out-of-Distribution): Data that does not follow the training-data distribution and differs significantly from the examples seen during training.

- Near-OOD: OOD samples that are semantically or visually similar to the in-distribution data. These samples typically share many characteristics with ID data and are therefore more difficult to distinguish from ID samples. Example: CIFAR-10 as ID and CIFAR-100 as OOD.

- Far-OOD: OOD samples that differ substantially from the in-distribution data in semantics, appearance, or underlying data-generating process. These samples are generally easier to distinguish from ID samples. Example: CIFAR-10 as ID and MNIST as OOD.

- OOD Score: A scalar measure of confidence used for OOD detection. Higher values indicate ID samples; lower values indicate OOD samples.

- MSP (Maximum Softmax Probability): An OOD score defined as the maximum softmax probability of the teacher model. MSP is a baseline for OOD detection.
