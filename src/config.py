"""Centralized configuration loader for the Medical Document NLP Analyzer.

Loads settings from config/config.yaml and environment variables (.env).
Environment variables always take precedence for credentials and overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv

from src.utils import get_project_root, setup_logger

logger = setup_logger(__name__)

# Load .env file from project root if it exists
project_root = get_project_root()
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()  # Fallback to system env / default search


class Config:
    """Application configuration manager."""

    def __init__(self, config_yaml_path: Optional[Path] = None):
        self.project_root = get_project_root()
        self.config_path = config_yaml_path or (self.project_root / "config" / "config.yaml")
        self.yaml_data: Dict[str, Any] = self._load_yaml()

        # Operational & Document configurations
        doc_cfg = self.yaml_data.get("document_processing", {})
        self.supported_file_types: List[str] = doc_cfg.get("supported_file_types", [".pdf", ".txt"])
        
        # Max file size: env override or yaml
        env_max_size = os.getenv("MAX_FILE_SIZE_MB")
        self.max_file_size_mb: int = int(env_max_size) if env_max_size else doc_cfg.get("max_file_size_mb", 10)

        # LLM Configurations
        llm_cfg = self.yaml_data.get("llm", {})
        self.llm_provider: str = os.getenv("LLM_PROVIDER", llm_cfg.get("default_provider", "gemini")).lower().strip()
        
        env_timeout = os.getenv("LLM_TIMEOUT")
        self.llm_timeout: int = int(env_timeout) if env_timeout else int(llm_cfg.get("timeout_seconds", 30))
        self.temperature: float = float(llm_cfg.get("temperature", 0.0))
        self.max_output_tokens: int = int(llm_cfg.get("max_output_tokens", 1024))

        # Gemini
        gemini_cfg = llm_cfg.get("gemini", {})
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        self.gemini_model: str = os.getenv("GEMINI_MODEL", gemini_cfg.get("default_model", "gemini-2.5-flash"))

        # OpenAI
        openai_cfg = llm_cfg.get("openai", {})
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.openai_model: str = os.getenv("OPENAI_MODEL", openai_cfg.get("default_model", "gpt-4o-mini"))

        # Validation sections
        val_cfg = self.yaml_data.get("report_validation", {})
        self.required_report_sections: List[str] = val_cfg.get("required_sections", [
            "MEDICAL DOCUMENT SUMMARY",
            "Symptoms:",
            "Medical Conditions:",
            "Medications:",
            "Dosages:",
            "Tests:",
            "Procedures:",
            "Body Parts:",
            "Important Findings:",
            "Brief Summary:",
            "Potential Missing Information:",
            "Safety Note:",
        ])

        # NLP Vocabulary
        self.nlp_vocabulary: Dict[str, List[str]] = self.yaml_data.get("nlp_vocabulary", {})
        
        # Findings keywords
        self.finding_keywords: List[str] = self.yaml_data.get("findings", {}).get("keywords", [])

        # Negation triggers
        neg_cfg = self.yaml_data.get("negation", {})
        self.pre_negation_triggers: List[str] = neg_cfg.get("pre_negation_triggers", [])
        self.post_negation_triggers: List[str] = neg_cfg.get("post_negation_triggers", [])
        self.historical_triggers: List[str] = neg_cfg.get("historical_triggers", [])

    def _load_yaml(self) -> Dict[str, Any]:
        """Safely load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config YAML file not found at {self.config_path}. Using empty defaults.")
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading configuration YAML: {e}")
            return {}

    def get_prompt_template(self) -> str:
        """Load the medical summary prompt template from prompts/."""
        prompt_path = self.project_root / "prompts" / "medical_summary_prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template missing at: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def get_api_key_for_provider(self, provider: Optional[str] = None) -> Optional[str]:
        """Retrieve the API key for a specified or current provider."""
        target_provider = (provider or self.llm_provider).lower().strip()
        if target_provider == "gemini":
            return self.gemini_api_key
        elif target_provider == "openai":
            return self.openai_api_key
        return None

    def validate_provider_configured(self, provider: Optional[str] = None) -> tuple[bool, str]:
        """Validate if the required credentials exist for the target provider."""
        target = (provider or self.llm_provider).lower().strip()
        if target not in ["gemini", "openai"]:
            return False, f"Unsupported provider: '{target}'. Must be 'gemini' or 'openai'."
        
        key = self.get_api_key_for_provider(target)
        if not key or not key.strip():
            env_var = "GEMINI_API_KEY" if target == "gemini" else "OPENAI_API_KEY"
            return False, f"Missing API key for {target.capitalize()}. Please set {env_var} in .env file."
        
        return True, "Configured"


# Default singleton instance for convenient import across modules
config = Config()
