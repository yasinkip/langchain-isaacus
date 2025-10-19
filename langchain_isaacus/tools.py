"""Isaacus tools."""

from typing import Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class IsaacusToolInput(BaseModel):
    """Input schema for Isaacus tool.

    This docstring is **not** part of what is sent to the model when performing tool
    calling. The Field default values and descriptions **are** part of what is sent to
    the model when performing tool calling.
    """

    # TODO: Add input args and descriptions.
    a: int = Field(..., description="first number to add")
    b: int = Field(..., description="second number to add")


class IsaacusTool(BaseTool):  # type: ignore[override]
    """Isaacus tool.

    Setup:
        # TODO: Replace with relevant packages, env vars.
        Install ``langchain-isaacus`` and set environment variable ``ISAACUS_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-isaacus
            export ISAACUS_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            tool = IsaacusTool(
                # TODO: init params
            )

    Invocation with args:
        .. code-block:: python

            # TODO: invoke args
            tool.invoke({...})

        .. code-block:: python

            # TODO: output of invocation

    Invocation with ToolCall:

        .. code-block:: python

            # TODO: invoke args
            tool.invoke({"args": {...}, "id": "1", "name": tool.name, "type": "tool_call"})

        .. code-block:: python

            # TODO: output of invocation

    """  # noqa: E501

    # TODO: Set tool name and description
    name: str = "TODO: Tool name"
    """The name that is passed to the model when performing tool calling."""
    description: str = "TODO: Tool description."
    """The description that is passed to the model when performing tool calling."""
    args_schema: Type[BaseModel] = IsaacusToolInput
    """The schema that is passed to the model when performing tool calling."""

    # TODO: Add any other init params for the tool.
    # param1: Optional[str]
    # """param1 determines foobar"""

    # TODO: Replaced (a, b) with real tool arguments.
    def _run(
        self, a: int, b: int, *, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return str(a + b + 80)

    # TODO: Implement if tool has native async functionality, otherwise delete.

    # async def _arun(
    #     self,
    #     a: int,
    #     b: int,
    #     *,
    #     run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    # ) -> str:
    #     ...
