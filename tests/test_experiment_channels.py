import asyncio
import os
from time import sleep

import pytest

from transformer_discord_notifier.discord import DiscordClient


# ----------------------------------------------------------------------------


# TODO: tests ...
# add cleanup methods (delete channel/category?)


# ----------------------------------------------------------------------------


def test_experiment_channel_no_create_explicit(monkeypatch):
    token = os.getenv("DISCORD_TOKEN")
    channel = os.getenv("DISCORD_CHANNEL")
    create_channels = False  # os.getenv("DISCORD_CREATE_EXPERIMENT_CHANNEL") == "True"
    monkeypatch.delenv("DISCORD_CREATE_EXPERIMENT_CHANNEL", raising=False)
    monkeypatch.delenv("DISCORD_EXPERIMENT_CATEGORY", raising=False)
    monkeypatch.delenv("DISCORD_EXPERIMENT_NAME", raising=False)

    client = DiscordClient(
        token=token,
        channel=channel,
        create_experiment_channels=create_channels,
    )

    client.init()

    assert not client._experiment_create_channels

    client.quit()


def test_experiment_channel_no_create_no_envvars(monkeypatch):
    token = os.getenv("DISCORD_TOKEN")
    channel = os.getenv("DISCORD_CHANNEL")
    monkeypatch.delenv("DISCORD_CREATE_EXPERIMENT_CHANNEL", raising=False)
    monkeypatch.delenv("DISCORD_EXPERIMENT_CATEGORY", raising=False)
    monkeypatch.delenv("DISCORD_EXPERIMENT_NAME", raising=False)

    client = DiscordClient(
        token=token,
        channel=channel,
    )

    client.init()

    assert not client._experiment_create_channels
    assert client._experiment_category_channel_name is None
    assert client._experiment_channel_name is None

    client.quit()


def test_experiment_create_channel(monkeypatch):
    token = os.getenv("DISCORD_TOKEN")
    channel = os.getenv("DISCORD_CHANNEL")
    create_channels = True  # os.getenv("DISCORD_CREATE_EXPERIMENT_CHANNEL") == "True"
    experiment_category = os.getenv("DISCORD_EXPERIMENT_CATEGORY")
    experiment_name = os.getenv("DISCORD_EXPERIMENT_NAME")

    client = DiscordClient(
        token=token,
        channel=channel,
        create_experiment_channels=create_channels,
        experiment_category=experiment_category,
        experiment_name=experiment_name,
    )

    client.init()

    assert isinstance(client._discord_channel, int)
    assert client._experiment_category_channel.name == experiment_category
    assert client._experiment_create_channels

    # cleanup
    if create_channels:
        cat = client._find_or_create_category_channel(experiment_category)
        chan = client._find_or_create_text_channel(experiment_name, cat)
        asyncio.run_coroutine_threadsafe(chan.delete(), client.loop)
        sleep(1)
        asyncio.run_coroutine_threadsafe(cat.delete(), client.loop)
        sleep(1)

    client.quit()


# ----------------------------------------------------------------------------
