"""Rule-based negation and temporal/historical status detection.

Analyzes local sentence context around medical entities using trigger windows
and clause boundaries to determine whether an entity is PRESENT, NOT_PRESENT,
or HISTORICAL.
"""

import re
from typing import List, Optional, Tuple
from src.models import EntityStatus
from src.config import config
from src.utils import setup_logger

logger = setup_logger(__name__)

# Conjunction boundaries that terminate the scope of negation/history
CLAUSE_BOUNDARIES = [
    r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\byet\b",
    r"\bwhile\b", r";", r"\bexcept\b", r"\bapart from\b"
]

DEFAULT_PRE_NEGATION_TRIGGERS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bnegative\s+for\b",
    r"\bno\s+evidence\s+of\b",
    r"\bno\s+sign\s+of\b",
    r"\brules?\s+out\b",
    r"\bruled\s+out\b",
    r"\bfree\s+of\b",
    r"\bnever\s+had\b",
    r"\bunremarkable\s+for\b",
    r"\babsent\b",
]

DEFAULT_POST_NEGATION_TRIGGERS = [
    r"\bwas\s+negative\b",
    r"\bis\s+negative\b",
    r"\bruled\s+out\b",
    r"\bnot\s+observed\b",
    r"\babsent\b",
    r"\bdenied\b",
    r"\bwas\s+not\s+found\b",
]

DEFAULT_HISTORICAL_TRIGGERS = [
    r"\bhistory\s+of\b",
    r"\bpast\s+history\s+of\b",
    r"\bpast\s+history\b",
    r"\bpreviously\s+had\b",
    r"\bformer\b",
    r"\bprior\s+history\s+of\b",
    r"\bprior\s+history\b",
    r"\bdiagnosed\s+in\s+the\s+past\b",
    r"\bprevious\b",
]


class NegationDetector:
    """Detects negation and historical status for medical entities in sentence context."""

    def __init__(
        self,
        pre_negation_triggers: Optional[List[str]] = None,
        post_negation_triggers: Optional[List[str]] = None,
        historical_triggers: Optional[List[str]] = None,
        pre_window_words: int = 6,
        post_window_words: int = 4
    ):
        self.pre_negation_patterns = [
            re.compile(p, re.IGNORECASE) for p in (pre_negation_triggers or DEFAULT_PRE_NEGATION_TRIGGERS)
        ]
        self.post_negation_patterns = [
            re.compile(p, re.IGNORECASE) for p in (post_negation_triggers or DEFAULT_POST_NEGATION_TRIGGERS)
        ]
        self.historical_patterns = [
            re.compile(p, re.IGNORECASE) for p in (historical_triggers or DEFAULT_HISTORICAL_TRIGGERS)
        ]
        self.pre_window_words = pre_window_words
        self.post_window_words = post_window_words

    def _truncate_to_last_clause(self, pre_context: str) -> str:
        """Truncate pre-context so triggers beyond a clause boundary (e.g. 'but', ';') do not leak."""
        # Find the rightmost clause boundary in pre_context
        latest_split = 0
        for pattern in CLAUSE_BOUNDARIES:
            for match in re.finditer(pattern, pre_context, re.IGNORECASE):
                if match.end() > latest_split:
                    latest_split = match.end()
        if latest_split > 0:
            return pre_context[latest_split:]
        return pre_context

    def _truncate_to_first_clause(self, post_context: str) -> str:
        """Truncate post-context at the first clause boundary."""
        earliest_split = len(post_context)
        for pattern in CLAUSE_BOUNDARIES:
            match = re.search(pattern, post_context, re.IGNORECASE)
            if match and match.start() < earliest_split:
                earliest_split = match.start()
        return post_context[:earliest_split]

    def detect_status(self, entity_text: str, sentence: str) -> EntityStatus:
        """Determine whether the entity in the sentence is PRESENT, NOT_PRESENT, or HISTORICAL.
        
        Args:
            entity_text: The medical entity phrase (e.g., "fever", "chest pain")
            sentence: The sentence string containing the entity
            
        Returns:
            EntityStatus enum
        """
        if not sentence or not entity_text:
            return EntityStatus.PRESENT

        # Locate entity in sentence (case-insensitive)
        match = re.search(r"\b" + re.escape(entity_text) + r"\b", sentence, re.IGNORECASE)
        if not match:
            # Fallback to substring if word boundary fails
            idx = sentence.lower().find(entity_text.lower())
            if idx == -1:
                return EntityStatus.PRESENT
            start_idx = idx
            end_idx = idx + len(entity_text)
        else:
            start_idx = match.start()
            end_idx = match.end()

        # Extract pre-context and post-context
        raw_pre = sentence[:start_idx]
        raw_post = sentence[end_idx:]

        # Handle clause boundaries
        pre_clause = self._truncate_to_last_clause(raw_pre)
        post_clause = self._truncate_to_first_clause(raw_post)

        # Window words limit
        pre_words = pre_clause.split()
        if len(pre_words) > self.pre_window_words:
            pre_context = " ".join(pre_words[-self.pre_window_words:])
        else:
            pre_context = pre_clause

        post_words = post_clause.split()
        if len(post_words) > self.post_window_words:
            post_context = " ".join(post_words[:self.post_window_words])
        else:
            post_context = post_clause

        # 1. Check for Historical triggers in pre-context
        for pattern in self.historical_patterns:
            if pattern.search(pre_context):
                return EntityStatus.HISTORICAL

        # 2. Check for Pre-Negation triggers
        for pattern in self.pre_negation_patterns:
            if pattern.search(pre_context):
                return EntityStatus.NOT_PRESENT

        # 3. Check for Post-Negation triggers
        for pattern in self.post_negation_patterns:
            if pattern.search(post_context):
                return EntityStatus.NOT_PRESENT

        # Default to PRESENT
        return EntityStatus.PRESENT


# Default singleton instance
negation_detector = NegationDetector()
