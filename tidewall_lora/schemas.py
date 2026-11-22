"""Tidewall-native protocol; no OpenSecura risk or billing semantics."""
from typing import Annotated, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

PROFILE='tidewall.prompt-injection.1'
ENSEMBLE_PROFILE='tidewall.prompt-injection.ensemble-q8.1'


class Message(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    role: Literal['user']
    content: StrictStr

    @field_validator('content')
    @classmethod
    def nonempty(cls,value):
        if not value.strip(): raise ValueError('empty message')
        try: value.encode('utf-8')
        except UnicodeEncodeError: raise ValueError('invalid Unicode text') from None
        return value


class ClassifyRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    schema_version: Annotated[StrictInt,Field(ge=1,le=1)]
    profile: Literal['tidewall.prompt-injection.1','tidewall.prompt-injection.ensemble-q8.1']
    request_id: StrictStr
    event_kind: Literal['prompt']
    messages: Annotated[list[Message],Field(min_length=1,max_length=1)]
    timeout_ms: Annotated[StrictInt,Field(ge=1,le=10000)]

    @field_validator('request_id')
    @classmethod
    def canonical_uuid(cls,value):
        if str(UUID(value))!=value: raise ValueError('noncanonical request identity')
        return value
