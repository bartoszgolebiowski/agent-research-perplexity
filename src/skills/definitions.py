from __future__ import annotations

"""Skill definitions used by the research workflow."""

from .models import (
    DiscoverEntitiesOutput,
    ExtractDataOutput,
    FormulateQueryOutput,
    ProcessSearchResultsOutput,
    RefineQueryOutput,
    ValidateExtractionOutput,
    VerifySourceOutput,
)
from .base import SkillDefinition, SkillName


# ---------------------------------------------------------------------------
# ICP Intelligence System Skills
# ---------------------------------------------------------------------------

FORMULATE_QUERY_SKILL = SkillDefinition(
    name=SkillName.FORMULATE_QUERY,
    template_name="skills/formulate_query.j2",
    output_model=FormulateQueryOutput,
)

REFINE_QUERY_SKILL = SkillDefinition(
    name=SkillName.REFINE_QUERY,
    template_name="skills/refine_query.j2",
    output_model=RefineQueryOutput,
)

# Split skills for better reliability with mid-sized LLMs
VERIFY_SOURCE_SKILL = SkillDefinition(
    name=SkillName.VERIFY_SOURCE,
    template_name="skills/verify_source.j2",
    output_model=VerifySourceOutput,
)

EXTRACT_DATA_SKILL = SkillDefinition(
    name=SkillName.EXTRACT_DATA,
    template_name="skills/extract_data.j2",
    output_model=ExtractDataOutput,
)

VALIDATE_EXTRACTION_SKILL = SkillDefinition(
    name=SkillName.VALIDATE_EXTRACTION,
    template_name="skills/validate_extraction.j2",
    output_model=ValidateExtractionOutput,
)

DISCOVER_ENTITIES_SKILL = SkillDefinition(
    name=SkillName.DISCOVER_ENTITIES,
    template_name="skills/discover_entities.j2",
    output_model=DiscoverEntitiesOutput,
)

ALL_SKILLS = [
    FORMULATE_QUERY_SKILL,
    REFINE_QUERY_SKILL,
    VERIFY_SOURCE_SKILL,
    EXTRACT_DATA_SKILL,
    VALIDATE_EXTRACTION_SKILL,
    DISCOVER_ENTITIES_SKILL,
]
