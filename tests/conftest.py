import os
from typing import Dict, Any

import pytest

# _pytest.config.argparsing.Parser
# _pytest.monkeypatch.MonkeyPatch
# _pytest.fixtures.FixtureRequest

# see: https://docs.pytest.org/en/stable/example/simple.html#request-example
# TODO: might be possible to just skip all tests that required valid tokens


def pytest_addoption(parser):
    def environ_or_required(key: str) -> Dict[str, Any]:
        return (
            {"default": os.environ.get(key)}
            if key in os.environ
            else {"required": True}
        )

    def environ_or_deault(key: str, default: Any) -> Dict[str, Any]:
        return {"default": os.environ.get(key) if key in os.environ else default}

    parser.addoption(
        "--discord-token",
        type=str,
        help="discord bot token",
        **environ_or_required("DISCORD_TOKEN"),
    )
    parser.addoption(
        "--discord-channel",
        type=str,
        help="discord bot channel name or id",
        **environ_or_required("DISCORD_CHANNEL"),
    )
    parser.addoption(
        "--discord-experiment-create-channels",
        action="store_true",
        help="discord bot creates new channels for experiments",
        **environ_or_deault("DISCORD_CREATE_EXPERIMENT_CHANNEL", False),
    )
    parser.addoption(
        "--discord-experiment-category",
        type=str,
        help="discord bot experiment channel category name",
        **environ_or_deault("DISCORD_EXPERIMENT_CATEGORY", None),
    )
    parser.addoption(
        "--discord-experiment-name",
        type=str,
        help="discord bot experiment name (channel name)",
        **environ_or_deault("DISCORD_EXPERIMENT_NAME", None),
    )


@pytest.fixture(autouse=True)
def env_setup(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    monkeypatch.setenv("DISCORD_TOKEN", request.config.getoption("--discord-token"))
    monkeypatch.setenv("DISCORD_CHANNEL", request.config.getoption("--discord-channel"))
    monkeypatch.setenv(
        "DISCORD_CREATE_EXPERIMENT_CHANNEL",
        request.config.getoption("--discord-experiment-create-channels"),
    )
    monkeypatch.setenv(
        "DISCORD_EXPERIMENT_CATEGORY",
        request.config.getoption("--discord-experiment-category"),
    )
    monkeypatch.setenv(
        "DISCORD_EXPERIMENT_NAME",
        request.config.getoption("--discord-experiment-name").lower().replace(" ", "-"),
    )
