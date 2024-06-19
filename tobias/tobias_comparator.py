import torch
import torchvision
from scipy.ndimage import gaussian_filter1d


class AggregatorInterface:
    def aggregate(
        self, control_signal: torch.Tensor, document: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class CentroidAggregator(AggregatorInterface):
    def __init__(self, similarity_metric):
        self.similarity_metric = similarity_metric

    def aggregate(
        self, control_signal: torch.Tensor, document: torch.Tensor
    ) -> torch.Tensor:
        # Ensure inputs are floats
        control_signal = control_signal.float()
        document = document.float()
        # Find centroid
        centroid = self._get_centroid(control_signal)
        # Calculate similarity
        similarity = self.similarity_metric(centroid, document)
        return similarity

    def _get_centroid(self, control_signal: torch.Tensor) -> torch.Tensor:
        # Calculate the total number of vectors
        num_vectors = control_signal.shape[0]
        component_sums = torch.sum(control_signal, axis=0)
        # Compute the mean for each component
        component_means = component_sums / num_vectors
        # The component_means array now contains the mean for each component
        centroid = component_means
        return centroid


class SumOfSimilaritiesAggregator(AggregatorInterface):
    def __init__(self, similarity_metric):
        self.similarity_metric = similarity_metric

    def aggregate(
        self, control_signal: torch.Tensor, document: torch.Tensor
    ) -> torch.Tensor:
        # Ensure inputs are floats
        control_signal = control_signal.float()
        document = document.float()

        similarities = torch.zeros(
            (control_signal.shape[0], document.shape[0]),
            device=document.device,
            dtype=document.dtype,
        )
        for i in range(control_signal.shape[0]):
            similarities[i] = self.similarity_metric(control_signal[i], document)

        # Sum and normalize similarities
        similarity = torch.sum(similarities, dim=0) / control_signal.shape[0]

        return similarity

class NoiseAggregator(AggregatorInterface):
    def aggregate(self, control_signal: torch.Tensor, document: torch.Tensor) -> torch.Tensor:
        return torch.rand((document.shape[0],), device=document.device, dtype=document.dtype)
        return super().aggregate(control_signal, document)


def cosine_similarity_metric(
    vector: torch.Tensor, vectors: torch.Tensor
) -> torch.Tensor:
    dot_products = torch.matmul(vectors, vector.unsqueeze(1)).squeeze()
    vector_norms = torch.linalg.norm(vectors, axis=1)
    query_vector_norm = torch.linalg.norm(vector)
    cosine_similarities = dot_products / (vector_norms * query_vector_norm)
    return cosine_similarities


def dot_similarity_metric(vector: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    dot_products = torch.matmul(vectors, vector)
    return dot_products


class TobiasComparator:
    def __init__(
        self,
        aggregator: AggregatorInterface,
        smoothing: bool,
        smoothing_window: int,
        smoothing_sigma: float,
    ):
        self.aggregator = aggregator
        self.smoothing = smoothing
        self.smoothing_window = smoothing_window
        self.smoothing_sigma = smoothing_sigma

    def _gaussian_kernel1d(self, kernel_size, sigma):
        """Creates a 1D Gaussian kernel using the given kernel size and sigma."""
        x = torch.arange(kernel_size) - kernel_size // 2
        kernel = torch.exp(-(x**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()  # Normalize the kernel
        return kernel

    def _gaussian_blur1d(self, input_tensor, kernel_size, sigma):
        """Applies a 1D Gaussian blur to the input tensor."""
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        kernel = self._gaussian_kernel1d(kernel_size, sigma)
        kernel = kernel.to(input_tensor.device)
        kernel = kernel.view(1, 1, -1)  # Reshape for 1D convolution

        # Add batch and channel dimensions to input tensor
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

        # Apply the convolution
        blurred_tensor = torch.nn.functional.conv1d(input_tensor, kernel, padding=kernel_size // 2)

        return blurred_tensor.squeeze(0).squeeze(0)

    def _smooth(self, value: torch.Tensor) -> torch.Tensor:
        return self._gaussian_blur1d(value, self.smoothing_window, self.smoothing_sigma)

    def compare(
        self, control_signal: torch.Tensor, document: torch.Tensor
    ) -> torch.Tensor:
        similarity = self.aggregator.aggregate(control_signal, document)

        # Smooth using gaussian smoother
        if self.smoothing:
            similarity = self._smooth(similarity)

        return similarity


def comparator_factory(
    aggregator_type,
    similarity_metric_type,
    smoothing,
    smoothing_window,
    smoothing_sigma,
):
    aggregator_types = ["centroid", "sum", "noise"]
    similarity_metric_types = {
        "dot": dot_similarity_metric,
        "cosine": cosine_similarity_metric,
    }

    if (aggregator_type not in aggregator_types) or (
        similarity_metric_type not in similarity_metric_types.keys()
    ):
        raise NameError

    similarity_metric = similarity_metric_types[similarity_metric_type]

    if aggregator_type == "centroid":
        aggregator = CentroidAggregator(similarity_metric)
    elif aggregator_type == "sum":
        aggregator = SumOfSimilaritiesAggregator(similarity_metric)
    elif aggregator_type == "noise":
        aggregator = NoiseAggregator()

    return TobiasComparator(
        aggregator=aggregator,
        smoothing=smoothing,
        smoothing_window=smoothing_window,
        smoothing_sigma=smoothing_sigma,
    )
