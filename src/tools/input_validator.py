from __future__ import annotations

"""Input validation tool for the ICP Intelligence System."""

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from src.memory.models import ICPAnalysisInput


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, message: str, errors: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.errors = errors or []


class ValidationRequest(BaseModel):
    """Request model for input validation tool."""

    file_path: Optional[Path] = Field(
        default=None, description="Path to JSON file to validate"
    )
    json_content: Optional[str] = Field(
        default=None, description="JSON string content to validate"
    )


class ValidationResponse(BaseModel):
    """Response model from input validation tool."""

    success: bool = Field(..., description="Whether validation succeeded")
    analysis_input: Optional[ICPAnalysisInput] = Field(
        default=None, description="Parsed input if validation succeeded"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if validation failed"
    )
    errors: List[str] = Field(
        default_factory=list, description="List of specific validation errors"
    )


class InputValidatorClient:
    """Validates input JSON files against the ICP analysis schema."""

    @classmethod
    def from_env(cls) -> "InputValidatorClient":
        """Create validator client (no env config needed)."""
        return cls()

    def validate(self, request: ValidationRequest) -> ValidationResponse:
        """Validate input from file or JSON string."""
        try:
            if request.file_path:
                result = self.validate_file(request.file_path)
            elif request.json_content:
                result = self.validate_json(request.json_content)
            else:
                raise ValidationError(
                    "Either file_path or json_content must be provided"
                )

            return ValidationResponse(success=True, analysis_input=result)
        except ValidationError as e:
            return ValidationResponse(
                success=False, error_message=str(e), errors=e.errors
            )

    def validate_file(self, file_path: Path) -> ICPAnalysisInput:
        """
        Validate a JSON file and return the parsed input specification.

        Args:
            file_path: Path to the JSON input file

        Returns:
            Validated ICPAnalysisInput instance

        Raises:
            ValidationError: If validation fails
        """
        if not file_path.exists():
            raise ValidationError(f"Input file not found: {file_path}")

        if not file_path.suffix.lower() == ".json":
            raise ValidationError(f"Input file must be JSON: {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as e:
            raise ValidationError(f"Failed to read file: {e}")

        return self.validate_json(content, str(file_path))

    def validate_json(
        self, json_content: str, source_path: Optional[str] = None
    ) -> ICPAnalysisInput:
        """
        Validate JSON content and return the parsed input specification.

        Args:
            json_content: JSON string to validate
            source_path: Optional source file path for error messages

        Returns:
            Validated ICPAnalysisInput instance

        Raises:
            ValidationError: If validation fails
        """
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON syntax: {e}")

        try:
            input_spec = ICPAnalysisInput.model_validate(data)
        except PydanticValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            raise ValidationError(
                f"Schema validation failed with {len(errors)} error(s)", errors=errors
            )

        # Additional semantic validation
        self._validate_semantics(input_spec)

        # Store the source path
        if source_path:
            input_spec.input_file_path = source_path

        return input_spec

    def _validate_semantics(self, input_spec: ICPAnalysisInput) -> None:
        """Perform semantic validation beyond schema validation."""
        errors: List[str] = []

        # Check for at least one root node
        if not input_spec.root_nodes:
            errors.append("At least one root analysis node is required")

        # Check for duplicate node IDs
        all_nodes = input_spec.get_all_nodes()
        seen_ids = set()
        for node in all_nodes:
            if node.id in seen_ids:
                errors.append(f"Duplicate node ID: {node.id}")
            seen_ids.add(node.id)

        # Check that nodes with sub-tasks have descriptions
        for node in all_nodes:
            if node.sub_tasks and not node.description:
                errors.append(f"Node '{node.name}' has sub-tasks but no description")

        # Check extraction fields have valid data types
        valid_types = {"string", "number", "date", "currency", "list", "boolean"}
        for node in all_nodes:
            for field in node.extraction_fields:
                if field.data_type not in valid_types:
                    errors.append(
                        f"Node '{node.name}': Invalid data type '{field.data_type}' "
                        f"for field '{field.name}'"
                    )

        if errors:
            raise ValidationError(
                f"Semantic validation failed with {len(errors)} error(s)", errors=errors
            )
