import os
import time
from concurrent.futures import TimeoutError
from typing import List, Set

import discord

import pytest

from transformer_discord_notifier.discord import DiscordClient


# ----------------------------------------------------------------------------


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

    with pytest.raises((RuntimeError, TimeoutError)):
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


def test_valid_token_and_channel_combinations(monkeypatch: pytest.MonkeyPatch):
    token = os.getenv("DISCORD_TOKEN")
    channel = os.getenv("DISCORD_CHANNEL")
    client = DiscordClient(token=token, channel=channel)
    client.init()
    client.quit()

    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_CHANNEL", raising=False)
    client = DiscordClient(token=token, channel=channel)
    client.init()
    client.quit()

    monkeypatch.setenv("DISCORD_TOKEN", token)
    client = DiscordClient(token=None, channel=channel)
    client.init()
    client.quit()

    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.setenv("DISCORD_CHANNEL", channel)
    client = DiscordClient(token=token, channel=None)
    client.init()
    client.quit()


# ----------------------------------------------------------------------------


def test_send_message_not_initialized(monkeypatch):
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_CHANNEL", raising=False)

    client = DiscordClient()

    with pytest.raises(RuntimeError) as exc_info:
        client.init()
    assert exc_info.type is RuntimeError
    assert exc_info.value.args[0] == "No DISCORD_TOKEN environment variable set!"

    assert client._initialized is False

    ret = client.send_message("test")
    assert ret is None

    ret = client.send_message()
    assert ret is None

    time.sleep(0.1)
    client.quit()


def test_send_message_and_edit_and_delete_if_initialized():
    client = DiscordClient()
    client.init()
    assert client._initialized is True

    # send message
    msg_id = client.send_message("test")
    assert msg_id is not None
    # update
    msg_id2 = client.update_or_send_message(msg_id=msg_id, text="test2")
    assert msg_id2 == msg_id
    # clean up
    ret = client.delete_later(msg_id, delay=0)
    assert ret is True

    # send message via update on non-existent message
    msg_id3 = client.update_or_send_message(msg_id=None, text="test3")
    assert msg_id3 is not None
    # clean up
    ret = client.delete_later(msg_id3, delay=0)
    assert ret is True

    time.sleep(0.5)
    ret = client.delete_later(msg_id3, delay=0)
    assert ret is False

    # invalid input, should only be int
    with pytest.raises(discord.errors.HTTPException):
        client.delete_later(None, delay=0)

    # should not send anything as nothing is being provided
    ret = client.send_message()
    assert ret is None

    time.sleep(0.1)
    client.quit()


def test_empty_sending_if_initialized():
    client = DiscordClient()
    client.init()
    assert client._initialized is True

    # should not send anything as nothing is being provided
    ret = client.send_message()
    assert ret is None

    # try to update without content
    ret = client.update_or_send_message(msg_id=None)
    assert ret is None

    # send message
    msg_id = client.send_message("test")
    assert msg_id is not None
    # update with nothing
    msg_id2 = client.update_or_send_message(msg_id=msg_id)
    assert msg_id2 == msg_id
    # check message content
    msg = client.get_message_by_id(msg_id)
    assert msg is not None
    assert msg.content == "test"
    # clean up
    ret = client.delete_later(msg_id, delay=0)
    assert ret is True

    # wait for deletion to happen ...
    time.sleep(1)
    # check message does not exist anymore
    msg = client.get_message_by_id(msg_id)
    assert msg is None
    # TODO: alternatively loop until deleted, for n seconds/iterations

    client.quit()


def test_embed_if_not_initialized(monkeypatch):
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_CHANNEL", raising=False)

    client = DiscordClient()
    with pytest.raises(RuntimeError):
        client.init()
    assert client._initialized is False

    embed = client.build_embed(kvs=dict(), title="test")
    embed_d = embed.to_dict()
    assert embed_d == {"type": "rich", "title": "test"}

    embed2 = DiscordClient.build_embed(dict(), title="test")
    embed_d2 = embed2.to_dict()
    assert embed_d2 == embed_d

    embed3 = client.build_embed(kvs={"a": "b", "c": 1, "d": 1.2}, footer="t2")
    embed_d3 = embed3.to_dict()
    assert embed_d3 == {
        "type": "rich",
        "footer": {"text": "t2"},
        "fields": [
            # text will be formatted as json via pprint.pformat
            {"inline": False, "name": "a", "value": "```json\n'b'\n```"},
            # numbers normal
            {"inline": True, "name": "c", "value": "1"},
            # floats too
            {"inline": True, "name": "d", "value": "1.2"},
        ],
    }

    ret = client.send_message(embed=embed)
    assert ret is None

    ret = client.send_message(text="test1", embed=embed)
    assert ret is None

    client.quit()


def test_embed_if_initialized():
    client = DiscordClient()
    client.init()
    assert client._initialized is True

    # create embed
    embed = client.build_embed(kvs=dict(), title="test")
    embed_d = embed.to_dict()
    assert embed_d == {"type": "rich", "title": "test"}
    # send embed
    msg_id = client.send_message(embed=embed)
    assert msg_id is not None
    # clean up
    ret = client.delete_later(msg_id, delay=0)
    assert ret is True

    embed = client.build_embed(
        kvs={"a": "b", "c": 1, "d": 1.2}, title="title", footer="t2"
    )
    # send message but without message content (so, only the embed)
    msg_id = client.send_message(embed=embed)
    assert msg_id is not None

    # check message that was sent (?) or is in client cache
    msg = client.get_message_by_id(msg_id)
    assert msg is not None
    assert len(msg.embeds) == 1
    assert msg.content == ""

    # try to remove embed but without text in message
    msg_id2 = client.update_or_send_message(msg_id, embed=None)
    assert msg_id2 == msg_id
    # should keep the embed message without changing anything
    msg2 = client.get_message_by_id(msg_id2)
    assert msg2 is not None
    assert len(msg2.embeds) == 1
    assert msg2.content == ""
    # same message but different instances
    assert msg == msg2
    assert id(msg) != id(msg2)

    # remove embed, but update/add text
    msg_id3 = client.update_or_send_message(msg_id, text="t", embed=None)
    assert msg_id3 == msg_id
    # embed should have vanished
    msg3 = client.get_message_by_id(msg_id3)
    assert msg3 is not None
    assert len(msg3.embeds) == 0
    assert msg3.content == "t"
    assert msg == msg3
    # clean up
    ret = client.delete_later(msg_id, delay=0)
    assert ret is True

    time.sleep(0.1)
    client.quit()


# ----------------------------------------------------------------------------


def test_calls_if_not_initialized():
    client = DiscordClient()
    assert client._initialized is False

    ret = client.send_message("test")
    assert ret is None

    ret = client.get_message_by_id(1)
    assert ret is None

    ret = client.update_or_send_message(1)
    assert ret is None

    ret = client.delete_later(1, 2)
    assert ret is False

    ret = client.build_embed(kvs=dict(), title="test")
    assert ret is not None

    client._quit_client()
    client.quit()


def test_channel_finding():
    token = os.getenv("DISCORD_TOKEN")
    channel = os.getenv("DISCORD_CHANNEL")
    client = DiscordClient(token=token, channel=channel)
    client.init()
    assert client._initialized is True

    # check that channels exist and get all text channels
    channels: List[discord.abc.GuildChannel] = list(client.client.get_all_channels())
    channels = [ch for ch in channels if ch.type == discord.ChannelType.text]
    channels = [ch for ch in channels if ch.permissions_for(ch.guild.me).send_messages]
    channel_names: Set[str] = {ch.name for ch in channels}
    assert len(channels) > 0, "testing requires access to text channels"

    # search for unique/new channel name
    # not sure if this can break because of max length limits ...
    new_channel_name: str = channels[0].name + "A"
    while new_channel_name in channel_names:
        new_channel_name = new_channel_name + "A"

    # trying to find a channel by name that we do not have access to
    # should raise an error
    with pytest.raises(RuntimeError) as exc_info:
        client._discord_channel = new_channel_name
        client._find_default_channel(
            name=new_channel_name, default_name=new_channel_name
        )
    assert exc_info.type is RuntimeError

    # 4th try
    client._discord_channel = new_channel_name
    channel_id = client._find_default_channel(
        name=new_channel_name, default_name=channel
    )
    assert isinstance(channel_id, int)

    # 3rd try
    client._discord_channel = channel_id
    channel_id2 = client._find_default_channel(
        name=new_channel_name, default_name=new_channel_name
    )
    assert channel_id2 == channel_id

    # 2nd try
    client._discord_channel = channel
    channel_id3 = client._find_default_channel(
        name=new_channel_name, default_name=new_channel_name
    )
    assert channel_id3 == channel_id

    # 1st try
    client._discord_channel = new_channel_name
    channel_id4 = client._find_default_channel(
        name=channel, default_name=new_channel_name
    )
    assert channel_id4 == channel_id

    # should not work with ids as name
    client._discord_channel = new_channel_name
    with pytest.raises(RuntimeError) as exc_info:
        client._find_default_channel(name=channel_id, default_name=new_channel_name)

    # should not work with ids as name
    client._discord_channel = new_channel_name
    with pytest.raises(RuntimeError) as exc_info:
        client._find_default_channel(name=new_channel_name, default_name=channel_id)

    # TODO: hashtag channel names!

    client.quit()


def test_cancel_tasks():
    client = DiscordClient()
    client.init()
    assert client._initialized is True

    msg_id = client.send_message("**TODO:** delete test message manually!")
    assert msg_id is not None

    found = client.delete_later(msg_id, delay=10)
    assert found is True

    client.quit()

    # ---

    client = DiscordClient()
    client.init()
    found = client.delete_later(msg_id, delay=0)
    assert found is True
    time.sleep(0.1)
    client.quit()


# ----------------------------------------------------------------------------
