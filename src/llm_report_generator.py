"""LLM Report Generator that strictly feeds validated structured data to Cloud LLM.

Never sends raw clinical documents or unverified patient text to the model.
Includes automatic fallback to deterministic rule-based report generator upon
any API error, timeout, or validation failure.
"""

from typing import Optional, Tuple
import json

from src.models import ExtractedMedicalData, ReportValidationResult
from src.llm_client import BaseLLMClient, get_llm_client, LLMClientError
from src.report_validator import report_validator
from src.report_formatter import generate_rule_based_fallback_report
from src.config import config
from src.utils import setup_logger

logger = setup_logger(__name__)


class LLMReportGenerator:
    """Orchestrates structured data prompt construction, LLM generation, and validation."""

    def __init__(self, client: Optional[BaseLLMClient] = None, prompt_template: Optional[str] = None):
        self.client = client
        self.prompt_template = prompt_template or config.get_prompt_template()

    def _build_prompt(self, data: ExtractedMedicalData) -> str:
        """Inject ONLY structured extracted information into the prompt template.
        
        Raw document text is strictly excluded to prevent unverified extraction.
        """
        # Convert to readable structured bullet text
        structured_bullets = data.to_formatted_bullet_text()
        return self.prompt_template.format(structured_data=structured_bullets)

    def generate_report(
        self,
        data: ExtractedMedicalData,
        client: Optional[BaseLLMClient] = None
    ) -> Tuple[str, bool, Optional[str], Optional[ReportValidationResult]]:
        """Generate a validated factual medical summary report.
        
        Args:
            data: Structured ExtractedMedicalData from NLP pipeline
            client: Optional LLM client instance (defaults to self.client or factory)
            
        Returns:
            Tuple of:
            - report_text: Final report string (LLM generated or Fallback)
            - is_fallback: Boolean True if fallback was triggered
            - error_message: Error description if LLM failed, else None
            - validation_result: ReportValidationResult object
        """
        active_client = client or self.client
        if active_client is None:
            try:
                active_client = get_llm_client()
            except Exception as e:
                logger.warning(f"Could not initialize default LLM client: {e}. Using fallback report.")
                fallback = generate_rule_based_fallback_report(data, failure_reason=str(e))
                val_res = report_validator.validate(fallback, data)
                return fallback, True, f"LLM client setup error: {str(e)}", val_res

        # If document had zero entities at all
        if data.total_entities_count() == 0:
            logger.info("Zero entities extracted. Generating fallback report directly.")
            fallback = generate_rule_based_fallback_report(data, failure_reason="No entities detected in document")
            val_res = report_validator.validate(fallback, data)
            return fallback, True, "No medical entities were detected in the document.", val_res

        prompt = self._build_prompt(data)

        # Attempt Cloud LLM generation
        try:
            logger.info(f"Sending structured data to {active_client.provider_name} ({active_client.model_name})")
            llm_output = active_client.generate(prompt)

            # Validate generated output
            validation = report_validator.validate(llm_output, data)
            if not validation.is_valid:
                logger.warning(
                    f"LLM output failed validation: {validation.warnings}. Falling back to rule-based report."
                )
                fallback = generate_rule_based_fallback_report(
                    data,
                    failure_reason=f"LLM output failed structural validation: {', '.join(validation.warnings)}"
                )
                return fallback, True, f"LLM output validation failed: {', '.join(validation.warnings)}", validation

            # Generation and validation succeeded
            return llm_output, False, None, validation

        except LLMClientError as e:
            logger.warning(f"LLM API error ({type(e).__name__}): {e}. Activating fallback.")
            fallback = generate_rule_based_fallback_report(data, failure_reason=str(e))
            val_res = report_validator.validate(fallback, data)
            return fallback, True, str(e), val_res

        except Exception as e:
            logger.error(f"Unexpected error in report generator: {e}. Activating fallback.")
            fallback = generate_rule_based_fallback_report(data, failure_reason=f"Unexpected error: {str(e)}")
            val_res = report_validator.validate(fallback, data)
            return fallback, True, f"Unexpected error during generation: {str(e)}", val_res


# Default singleton instance
llm_report_generator = LLMReportGenerator()
