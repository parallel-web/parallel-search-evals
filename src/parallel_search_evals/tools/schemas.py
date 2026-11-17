"""Parallel tool declarations, validated inputs, invocation and reply types."""

import json
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints


class ToolChoice(StrEnum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class ToolInvocation(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str = Field(strict=True, min_length=1, pattern=r"[^\s\x1c-\x1f]")
    name: str = Field(strict=True, min_length=1, pattern=r"[^\s\x1c-\x1f]")
    input: dict[str, Any]


class ToolReply(BaseModel):
    tool_call_id: str
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class ToolSpec:
    """Pair the committed provider schema with independent local argument validation.

    ``parameters`` preserves the manual schema rather than generating it from
    ``input_schema``. Provider adapters copy the nested parameters when building
    native requests so those projections cannot mutate the trusted declaration.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    input_schema: type[BaseModel]
    endpoint: str


SCHEMAS = json.loads(files(__package__).joinpath("assets/parallel-tools.json").read_text())
Objective = Annotated[str, StringConstraints(min_length=1, pattern=r"[^\s\x1c-\x1f]")]
Query = Annotated[str, StringConstraints(min_length=1, pattern=r"[^\s\x1c-\x1f]")]


def absolute_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Expected an absolute HTTP URL")
    return value


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: Objective
    search_queries: list[Query] = Field(min_length=1)


class ExtractInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: Objective
    urls: list[Annotated[str, AfterValidator(absolute_url)]] = Field(min_length=1)
    search_queries: list[Annotated[str, StringConstraints(min_length=1, pattern=r"[^\s\x1c-\x1f]")]] = Field(
        default_factory=list
    )


SEARCH = ToolSpec(SCHEMAS[0]["name"], SCHEMAS[0]["description"], SCHEMAS[0]["parameters"], SearchInput, "search")
EXTRACT = ToolSpec(SCHEMAS[1]["name"], SCHEMAS[1]["description"], SCHEMAS[1]["parameters"], ExtractInput, "extract")
TOOLS = (SEARCH, EXTRACT)
