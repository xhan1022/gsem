"""Configuration management for GSEM Pipeline."""
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class DeepSeekConfig(BaseModel):
    """DeepSeek API configuration."""
    api_key: str = Field(..., description="DeepSeek API key")
    base_url: str = Field(default="https://api.deepseek.com", description="DeepSeek API base URL")
    model_name: str = Field(default="deepseek-chat", description="Model name")
    temperature: float = Field(default=0.8, description="Sampling temperature")


class OpenAIConfig(BaseModel):
    """OpenAI API configuration for Graph Construction."""
    api_key: str = Field(..., description="OpenAI API key")
    base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    model_name: str = Field(default="gpt-4o-mini", description="Model name")
    temperature: float = Field(default=0.0, description="Sampling temperature")


class EmbeddingConfig(BaseModel):
    """Embedding API configuration."""
    api_key: str = Field(..., description="Embedding API key")
    base_url: str = Field(default="https://api.openai.com/v1", description="Embedding API base URL")
    model_name: str = Field(default="text-embedding-3-small", description="Embedding model name")


class PipelineConfig(BaseModel):
    """Pipeline configuration."""
    sampling_count: int = Field(default=4, description="Number of trajectory samples per case")
    max_workers: int = Field(default=5, description="Max parallel workers")
    compression_ratio: tuple[float, float] = Field(default=(0.3, 0.4), description="Target compression ratio for normalization")
    erv_threshold: float = Field(default=0.6, description="Acceptable performance threshold (tau) for ERV quality assessment")


class PathConfig(BaseModel):
    """Data path configuration."""
    data_dir: str = Field(default="data", description="Base data directory")

    # Dataset configuration
    train_dataset_path: str = Field(..., description="Direct path to training dataset file")
    test_dataset_path: str = Field(..., description="Direct path to test dataset file")
    dataset_split: str = Field(default="train", description="Dataset split: 'train' or 'test'")

    @property
    def dataset_path(self) -> str:
        """Get the full path to the dataset file based on dataset_split."""
        if self.dataset_split == "train":
            return self.train_dataset_path
        elif self.dataset_split == "test":
            return self.test_dataset_path
        else:
            raise ValueError(f"Invalid dataset_split: {self.dataset_split}. Must be 'train' or 'test'.")

    @property
    def cases_dir(self) -> str:
        return os.path.join(self.data_dir, "cases")

    @property
    def trajectories_dir(self) -> str:
        return os.path.join(self.data_dir, "trajectories")

    @property
    def evaluations_dir(self) -> str:
        return os.path.join(self.data_dir, "evaluations")

    @property
    def normalized_dir(self) -> str:
        return os.path.join(self.data_dir, "normalized")

    @property
    def positive_knowledge_dir(self) -> str:
        return os.path.join(self.data_dir, "positive_knowledge")

    @property
    def failure_analysis_dir(self) -> str:
        return os.path.join(self.data_dir, "failure_analysis")

    @property
    def experiences_dir(self) -> str:
        return os.path.join(self.data_dir, "experiences")

    @property
    def logs_dir(self) -> str:
        return "logs"


class Config:
    """Global configuration."""

    def __init__(self):
        self.deepseek = DeepSeekConfig(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model_name=os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat"),
            temperature=float(os.getenv("TEMPERATURE", "0.8"))
        )

        self.openai = OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            temperature=0.0
        )

        self.embedding = EmbeddingConfig(
            api_key=os.getenv("EMBEDDING_API_KEY", ""),
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )

        self.pipeline = PipelineConfig(
            sampling_count=int(os.getenv("SAMPLING_COUNT", "4")),
            max_workers=int(os.getenv("MAX_WORKERS", "5")),
            erv_threshold=float(os.getenv("ERV_THRESHOLD", "0.6"))
        )

        self.paths = PathConfig(
            data_dir=os.getenv("DATA_DIR", "data"),
            train_dataset_path=os.getenv("TRAIN_DATASET_PATH", "data/backup/MedRbench.json"),
            test_dataset_path=os.getenv("TEST_DATASET_PATH", "data/backup/MedRbench_test.json"),
            dataset_split=os.getenv("DATASET_SPLIT", "train")
        )

    def validate(self) -> bool:
        """Validate configuration."""
        if not self.deepseek.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        if not self.openai.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not self.embedding.api_key:
            raise ValueError("EMBEDDING_API_KEY is required")

        return True


# Global config instance
config = Config()
