from concurrent.futures import TimeoutError

import pytest

from transformer_discord_notifier.discord import DiscordClient


def test_no_token_channel(monkeypatch):
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_CHANNEL", raising=False)

    client = DiscordClient(token=None)

    with pytest.raises(RuntimeError) as exc_info:
        client.init()
    assert exc_info.type is RuntimeError
    assert exc_info.value.args[0] == "No DISCORD_TOKEN environment variable set!"

    client.quit()


def test_invalid_token(monkeypatch):
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_CHANNEL", raising=False)

    client = DiscordClient(token="1")

    with pytest.raises((RuntimeError, TimeoutError)) as exc_info:
        client.init()

    assert client._initialized is False

    # assert exc_info.type is RuntimeError
    # in log output: "Login error! Improper token has been passed."

    client.quit()


def test_valid_token_no_channel(monkeypatch):
    monkeypatch.delenv("DISCORD_CHANNEL", raising=False)

    client = DiscordClient()

    with pytest.raises(RuntimeError) as exc_info:
        client.init()
    assert exc_info.type is RuntimeError
    assert exc_info.value.args[0] == "No Text channel found!"
    # NOTE: this might work depending on the localized discord bot language
    # and default guild/server channel names?

    client.quit()


def test_valid_token_and_channel(monkeypatch):
    client = DiscordClient()
    client.init()
    client.quit()
