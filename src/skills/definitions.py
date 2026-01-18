from __future__ import annotations

"""Skill definitions used by the research workflow."""

from .models import (
    DiscoverEntitiesOutput,
    ExtractDataOutput,
    FormulateQueryOutput,
    RefineQueryOutput,
    ValidateDataOutput,
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

EXTRACT_DATA_SKILL = SkillDefinition(
    name=SkillName.EXTRACT_DATA,
    template_name="skills/extract_data.j2",
    output_model=ExtractDataOutput,
)

VALIDATE_DATA_SKILL = SkillDefinition(
    name=SkillName.VALIDATE_DATA,
    template_name="skills/validate_data.j2",
    output_model=ValidateDataOutput,
)

REFINE_QUERY_SKILL = SkillDefinition(
    name=SkillName.REFINE_QUERY,
    template_name="skills/refine_query.j2",
    output_model=RefineQueryOutput,
)

DISCOVER_ENTITIES_SKILL = SkillDefinition(
    name=SkillName.DISCOVER_ENTITIES,
    template_name="skills/discover_entities.j2",
    output_model=DiscoverEntitiesOutput,
)

ALL_SKILLS = [
    FORMULATE_QUERY_SKILL,
    EXTRACT_DATA_SKILL,
    VALIDATE_DATA_SKILL,
    REFINE_QUERY_SKILL,
    DISCOVER_ENTITIES_SKILL,
]
