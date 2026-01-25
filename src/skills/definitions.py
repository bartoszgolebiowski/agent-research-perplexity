from __future__ import annotations

"""Skill definitions used by the research workflow."""

from .models import (
    FormulateQueryOutput,
    ProcessSearchResultsOutput,
    RefineQueryOutput,
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

PROCESS_RESULTS_SKILL = SkillDefinition(
    name=SkillName.PROCESS_RESULTS,
    template_name="skills/process_results.j2",
    output_model=ProcessSearchResultsOutput,
)

ALL_SKILLS = [
    FORMULATE_QUERY_SKILL,
    REFINE_QUERY_SKILL,
    PROCESS_RESULTS_SKILL,
]
