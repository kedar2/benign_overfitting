import torch

# Data model for benign overfitting
class DataModel:
    def __init__(self, input_dim: int, num_samples: int, corruption_rate: float, signal_level: float=0.5, num_validation_samples: int=1000):
        self.input_dim = input_dim
        self.num_samples = num_samples
        self.num_validation_samples = num_validation_samples
        self.corruption_rate = corruption_rate
        self.signal_level = signal_level

        # Generates a signal vector of unit norm
        self.signal_vector = torch.randn(input_dim)
        self.signal_vector /= torch.norm(self.signal_vector)

        # Initialize the training dataset
        self.set_train_dataset()
        self.set_validation_dataset()
    def set_train_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly generates a training dataset.
        Each output y_i is uniformly sampled from {-1, 1}, then corrupted with probability corruption_rate.
        Each x_i is of the form
            x_i = sqrt(signal_level) * y_i * signal_vector + sqrt(1 - signal_level) * noise,
        where noise is a (Gaussian) random vector orthogonal to signal_vector.

        Returns input dataset X and output dataset y.
        """
        input_dataset = torch.zeros(self.num_samples, self.input_dim)
        output_dataset = torch.zeros(self.num_samples, dtype=torch.int)
        for i in range(self.num_samples):
            true_y = (2 * (torch.rand(1) < 0.5).float() - 1).int().item()
            observed_y = true_y if torch.rand(1).item() > self.corruption_rate else -true_y
            x = (self.signal_level ** 0.5) * true_y * self.signal_vector
            noise = torch.randn(self.input_dim) / self.input_dim ** 0.5
            noise -= torch.dot(noise, self.signal_vector) * self.signal_vector
            x += ((1 - self.signal_level) ** 0.5) * noise
            input_dataset[i] = x
            output_dataset[i] = 1 if observed_y == 1 else 0
        self.train_dataset = input_dataset, output_dataset.reshape(-1, 1)
        return self.train_dataset
    def train_error(self, model: torch.nn.Module) -> float:
        """
        Computes the error of the model on the training dataset.
        """
        X, y = self.train_dataset
        outputs = model(X)
        return 1 - (outputs > 0).int().eq(y).float().mean().item()
    def set_validation_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly generates a validation dataset.
        Each output y_i is uniformly sampled from {-1, 1}, then corrupted with probability corruption_rate.
        Each x_i is of the form
            x_i = sqrt(signal_level) * y_i * signal_vector + sqrt(1 - signal_level) * noise,
        where noise is a (Gaussian) random vector orthogonal to signal_vector.

        Returns input dataset X and output dataset y.
        """
        input_dataset = torch.zeros(self.num_validation_samples, self.input_dim)
        output_dataset = torch.zeros(self.num_validation_samples, dtype=torch.int)
        for i in range(self.num_samples):
            y = (2 * (torch.rand(1) < 0.5).float() - 1).int().item()
            x = (self.signal_level ** 0.5) * y * self.signal_vector
            noise = torch.randn(self.input_dim) / self.input_dim ** 0.5
            noise -= torch.dot(noise, self.signal_vector) * self.signal_vector
            x += ((1 - self.signal_level) ** 0.5) * noise
            input_dataset[i] = x
            output_dataset[i] = 1 if y == 1 else 0
        self.validation_dataset = input_dataset, output_dataset.reshape(-1, 1)
        return self.validation_dataset
    def generalization_error(self, model: torch.nn.Module) -> float:
        """
        Computes the error of the model on a new dataset.
        """
        X, y = self.validation_dataset
        outputs = model(X)
        return 1 - (outputs > 0).int().eq(y).float().mean().item()