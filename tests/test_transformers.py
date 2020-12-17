import os
import time
from typing import Optional, Union

import discord
from transformers.trainer_callback import TrainerControl
from transformers.trainer_callback import TrainerState
from transformers.training_args import TrainingArguments

import pytest

from transformer_discord_notifier.discord import DiscordClient
from transformer_discord_notifier.transformers import DiscordProgressCallback


# ----------------------------------------------------------------------------


def test_minimal_workflow(monkeypatch):
    token = os.getenv("DISCORD_TOKEN")
    channel = os.getenv("DISCORD_CHANNEL")

    dpc = DiscordProgressCallback()
    dpc.start()
    assert dpc.disabled is False
    dpc.end()
    assert dpc.disabled is True

    dpc = DiscordProgressCallback(token=token, channel=channel)
    dpc.start()
    assert dpc.disabled is False
    dpc.end()

    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_CHANNEL", raising=False)

    dpc = DiscordProgressCallback()
    dpc.start()
    assert dpc.disabled is True
    dpc.end()

    dpc = DiscordProgressCallback(token=token, channel=channel)
    dpc.start()
    assert dpc.disabled is False
    dpc.end()


def test_client_attr():
    dpc = DiscordProgressCallback()
    assert dpc.disabled is True
    assert dpc.client is not None
    assert dpc.client._initialized is False
    dpc.start()
    assert dpc.disabled is False
    assert dpc.client._initialized is True
    dpc.end()
    assert dpc.disabled is True
    assert dpc.client is not None
    assert dpc.client._initialized is False


def test_del(monkeypatch):
    dpc = DiscordProgressCallback()
    dpc.start()
    assert dpc.disabled is False

    client = dpc.client
    assert client._initialized is True

    del dpc
    assert client._initialized is False


def test_trainer_workflow(monkeypatch: pytest.MonkeyPatch):
    args = TrainingArguments(output_dir=None)
    state = TrainerState()
    control = TrainerControl()
    kwargs = dict()

    eval_dataloader = [None] * 10
    logs = {"metric": 1}

    num_epochs = 2
    num_steps_per_epoch = 3

    dpc = DiscordProgressCallback()
    dpc.start()
    assert dpc.disabled is False
    # same as start()
    dpc.on_init_end(args, state, control, **kwargs)

    # --------------------------------
    # do some mocking here ...
    all_messages = set()
    all_messages_deleted = set()
    all_messages_embed = set()

    def mock_send_message(
        text: str = "", embed: Optional[discord.Embed] = None
    ) -> Optional[int]:
        msg_id = len(all_messages) + 1
        all_messages.add(msg_id)
        if embed is not None:
            all_messages_embed.add(msg_id)
        return msg_id

    def mock_update_or_send_message(
        msg_id: Optional[int] = None, **fields
    ) -> Optional[int]:
        if msg_id not in (all_messages - all_messages_deleted):
            msg_id = len(all_messages) + 1
            all_messages.add(msg_id)
        return msg_id

    def mock_delete_later(msg_id: int, delay: Union[int, float] = 5) -> bool:
        assert msg_id in all_messages
        all_messages_deleted.add(msg_id)
        return True

    monkeypatch.setattr(dpc.client, "send_message", mock_send_message)
    monkeypatch.setattr(
        dpc.client, "update_or_send_message", mock_update_or_send_message
    )
    monkeypatch.setattr(dpc.client, "delete_later", mock_delete_later)

    # --------------------------------
    # fake training
    # NOTE: training workflow should be something like this

    dpc.on_train_begin(args, state, control, **kwargs)
    # run for N epochs
    for i in range(num_epochs):
        dpc.on_epoch_begin(args, state, control, **kwargs)
        # with each M steps
        for j in range(num_steps_per_epoch):
            state.epoch = i + j / num_steps_per_epoch
            dpc.on_step_end(args, state, control, **kwargs)

            # fake evaluation run each 5 steps
            if i > 0 and (i * num_steps_per_epoch + j) % 5 == 0:
                for _ in range(len(eval_dataloader)):
                    dpc.on_prediction_step(
                        args, state, control, eval_dataloader=eval_dataloader
                    )
                dpc.on_evaluate(args, state, control)
                dpc.on_log(args, state, control, logs=logs)
        dpc.on_epoch_end(args, state, control, **kwargs)
    dpc.on_train_end(args, state, control, **kwargs)
    dpc.on_log(args, state, control, logs=logs)

    for _ in range(len(eval_dataloader)):
        dpc.on_prediction_step(args, state, control, eval_dataloader=eval_dataloader)
    dpc.on_evaluate(args, state, control)
    dpc.on_log(args, state, control, logs=logs)

    # --------------------------------

    # msgs = set(dpc.client.all_message_ids)

    dpc.end()
    assert dpc.disabled is True

    # msg.1 = train progress
    # msg.2 = train-eval progress
    # msg.3 = train-eval log
    # msg.4 = train log
    # msg.5 = eval progress
    # msg.6 = eval log

    # msg#progress = 1,2,5
    # msg#progress/eval = 2,5 (deleted in on_evaluate)
    # msg#embed = 3,4,6

    assert len(all_messages_deleted) == 2
    assert len(all_messages - all_messages_deleted) == 4
    assert len(all_messages_embed) == 3


def test_trainer_workflow_disabled(monkeypatch):
    args = TrainingArguments(output_dir=None)
    state = TrainerState()
    control = TrainerControl()
    kwargs = dict()

    eval_dataloader = [None] * 10
    logs = {"metric": 1}

    dpc = DiscordProgressCallback()
    assert dpc.disabled is True

    dpc.on_train_begin(args, state, control, **kwargs)
    dpc.on_epoch_begin(args, state, control, **kwargs)
    dpc.on_step_end(args, state, control, **kwargs)
    dpc.on_epoch_end(args, state, control, **kwargs)
    dpc.on_train_end(args, state, control, **kwargs)
    dpc.on_log(args, state, control, logs=logs)
    dpc.on_prediction_step(args, state, control, eval_dataloader=eval_dataloader)
    dpc.on_evaluate(args, state, control)
    dpc.on_log(args, state, control, logs=logs)

    assert len(dpc.client.all_message_ids) == 0

    dpc.end()
    assert dpc.disabled is True


def test_trainer_workflow_not_local_process_zero(monkeypatch):
    args = TrainingArguments(output_dir=None)
    state = TrainerState()
    control = TrainerControl()
    kwargs = dict()

    eval_dataloader = [None] * 10
    logs = {"metric": 1}

    state.is_local_process_zero = False
    state.is_world_process_zero = False

    dpc = DiscordProgressCallback()
    dpc.start()
    assert dpc.disabled is False

    dpc.on_train_begin(args, state, control, **kwargs)
    dpc.on_epoch_begin(args, state, control, **kwargs)
    dpc.on_step_end(args, state, control, **kwargs)
    dpc.on_epoch_end(args, state, control, **kwargs)
    dpc.on_train_end(args, state, control, **kwargs)
    dpc.on_log(args, state, control, logs=logs)
    dpc.on_prediction_step(args, state, control, eval_dataloader=eval_dataloader)
    dpc.on_evaluate(args, state, control)
    dpc.on_log(args, state, control, logs=logs)

    assert len(dpc.client.all_message_ids) == 0

    dpc.end()


# ----------------------------------------------------------------------------
