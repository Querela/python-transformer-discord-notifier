__version__ = "0.1.0"

import asyncio
import concurrent.futures
import logging
import os
import time
import threading
from datetime import timedelta
from functools import partial
from pprint import pformat

import discord
from discord.client import _cancel_tasks
from discord.client import _cleanup_loop

from transformers.trainer_callback import TrainerCallback
from transformers.trainer_callback import TrainerControl
from transformers.trainer_callback import TrainerState
from transformers.training_args import TrainingArguments

from typing import Optional, Union, Dict, Set, Any


LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------------


class MyClient(discord.Client):
    async def on_ready(self):
        LOGGER.debug("Logged on as {0}!".format(self.user))

    async def on_message(self, message):
        LOGGER.debug("Message from {0.author}: {0.content}".format(message))


# ----------------------------------------------------------------------------


class DiscordProgressCallback(TrainerCallback):
    def __init__(
        self, token: Optional[str] = None, channel: Optional[Union[str, int]] = None
    ):
        super().__init__()

        self._discord_token: Optional[str] = token
        self._discord_channel: Optional[Union[str, int]] = channel

        self.all_message_ids: Set[int] = set()
        self.last_message_ids: Dict[str, int] = dict()
        self.progressbars: Dict[str, int] = dict()
        self.timediffs: Dict[str, float] = dict()

        self._initialized: bool = False
        self.client_thread: Optional[threading.Thread] = None
        self.client: Optional[discord.Client] = None

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    # --------------------------------

    def _load_credentials(self):
        LOGGER.debug("Load credentials from env vars ...")
        if not self._discord_token:
            token = os.environ.get("DISCORD_TOKEN", None)
            if not token:
                raise RuntimeError("No DISCORD_TOKEN environment variable set!")
            self._discord_token = token

        if not self._discord_channel:
            channel = os.environ.get("DISCORD_CHANNEL", None)
            if channel:
                # TODO: try to strip leading '#'?
                try:
                    channel = int(channel)
                except ValueError:
                    pass
                self._discord_channel = channel

    def _find_default_channel(
        self, name: Optional[str] = None, default_name: str = "default"
    ) -> int:
        LOGGER.debug("Search for text channel to write to in Discord ...")

        guilds = self.client.guilds
        if not guilds:
            raise RuntimeError("No guilds found!")

        def serch_for_channel_by_name(
            name: str,
        ) -> Optional[discord.channel.TextChannel]:
            # all text channels where we can send messages
            text_channels = [
                channel
                for guild in guilds
                for channel in guild.channels
                if channel.type == discord.ChannelType.text
                and channel.permissions_for(guild.me).send_messages
            ]
            # only those with matching name
            text_channels = [
                channel for channel in text_channels if channel.name == name
            ]
            # sort which lowest position/id first (created first)
            text_channels = sorted(text_channels, key=lambda c: (c.position, c.id))
            if text_channels:
                return text_channels[0]
            return None

        channel = None

        # search by name if provided
        if name:
            channel = serch_for_channel_by_name(name)

        # search by envvar channel name if possible
        if not channel and isinstance(self._discord_channel, str):
            channel = serch_for_channel_by_name(self._discord_channel)

        # search by envvar channel id if possible
        if not channel and isinstance(self._discord_channel, int):
            try:
                channel = self.client.get_channel(self._discord_channel)
            except discord.errors.NotFound:
                channel = None

        # fall back to default channel names
        if not channel:
            channel = serch_for_channel_by_name(default_name)

        # fail
        if not channel:
            raise RuntimeError("No Text channel found!")

        return channel.id

    # --------------------------------

    def _init_discord(self):
        if self._initialized:
            LOGGER.debug("Already initialized, do nothing.")
            return

        self._load_credentials()

        self.client = MyClient(loop=self.loop)

        def client_thread_func():
            LOGGER.info(
                f"Running Discord AsyncIO Loop in Thread: {threading.current_thread()}"
            )
            asyncio.set_event_loop(self.loop)
            self.loop.create_task(self.client.start(self._discord_token))

            self.loop.run_forever()

        self.client_thread = threading.Thread(target=client_thread_func, daemon=True)
        self.client_thread.start()

        def search_text_channel():
            # LOGGER.debug("Wait for client to finish? ...")
            # await self.client.wait_until_ready()
            LOGGER.debug("Search for text channel ...")

            try:
                self._discord_channel = self._find_default_channel()
            except RuntimeError:
                LOGGER.warning("Found no default channel!")
                return None
            LOGGER.info(f"Found channel: {self._discord_channel}")
            return self._discord_channel

        # NOTE: that we have to set the loop in both the main and background thread!
        # else it will raise errors in Lock/Event classes ...
        # while not self.client.is_ready():
        #    pass
        future = asyncio.run_coroutine_threadsafe(
            self.client.wait_until_ready(), self.loop
        )
        _ = future.result()

        # th_future = concurrent.futures.Future()
        # def _async_wrap(func, fut):
        #    try:
        #        ret = func()
        #        fut.set_result(ret)
        #    except Exception as ex:
        #        print("ex", ex)
        #        fut.set_exception(ex)
        # self.loop.call_soon_threadsafe(
        #    _async_wrap,
        #    partial(
        #        asyncio.run_coroutine_threadsafe,
        #        self.client.wait_until_ready(),
        #        loop=self.loop,
        #    ),
        #    th_future,
        # )
        # th_future.result().result()

        channel_id = search_text_channel()
        print(channel_id)

        LOGGER.debug("Discord handler initialized.")

    def _quit_discord(self):
        if not self.client:
            LOGGER.debug("Discord already shutdown, do nothing.")
            return

        LOGGER.debug("Shutdown Discord handler ...")

        # stop client
        asyncio.run_coroutine_threadsafe(self.client.close(), self.loop)
        self.loop.close()
        self.client_thread.join(timeout=3)

        self.client = None
        self.client_thread = None

    # --------------------------------

    def _send_message(self, text: str) -> int:
        async def send():
            await self.client.wait_until_ready()

            channel = self.client.get_channel(self._discord_channel)
            msg = await channel.send(text)
            return msg

        future = asyncio.run_coroutine_threadsafe(send(), self.loop)
        message = future.result()
        # self.all_message_ids.add(message.id)
        return message.id

        # th_future = concurrent.futures.Future()
        # def _async_send(func, fut):
        #    try:
        #        ret = func()
        #        ret = ret.result()
        #        fut.set_result(ret)
        #    except Exception as ex:
        #        fut.set_exception(ex)
        # func = partial(asyncio.run_coroutine_threadsafe, send(), loop=self.loop)
        # self.loop.call_soon_threadsafe(_async_send, func, th_future)
        # message = th_future.result()

    def _delete_later(self, msg_id: int, delay: Union[int, float] = 5) -> bool:
        try:
            channel = self.client.get_channel(self._discord_channel)
            coro = channel.fetch_message(msg_id)
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            message = future.result()
        except discord.errors.NotFound:
            return False

        coro = message.delete(delay=delay)
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)

        return True

    def _send_log_results(
        self, logs: Dict[str, Any], state: TrainerState, args: TrainingArguments
    ) -> int:
        results_embed = discord.Embed(title="Results")

        for k, v in logs.items():
            if isinstance(v, (int, float)):
                results_embed.add_field(name=k, value=v, inline=True)
            else:
                results_embed.add_field(
                    name=k, value=f"```json\n{pformat(v)}\n```", inline=False
                )

        results_embed.set_footer(
            text=f"Global step: {state.global_step} | Run: {args.run_name}"
        )

        async def send():
            await self.client.wait_until_ready()

            channel = self.client.get_channel(self._discord_channel)
            msg = await channel.send(embed=results_embed)
            return msg

        future = asyncio.run_coroutine_threadsafe(send(), self.loop)
        message = future.result()
        self.all_message_ids.add(message.id)
        return message.id

    def _send_progress_msg(self, ptype: str, time_diff: Optional[float] = None) -> None:
        if ptype not in self.progressbars:
            return

        msg_id = self.last_message_ids.get(ptype, None)

        cur_step, max_step = self.progressbars[ptype]
        msg_s = (
            f"Progress [{ptype}]: "
            f"{round(cur_step / max_step * 100):d}% | "
            f"{cur_step} / {max_step}"
            + (f" | took: {timedelta(seconds=round(time_diff))!s}" if time_diff else "")
        )

        if msg_id:
            try:
                channel = self.client.get_channel(self._discord_channel)
                coro = channel.fetch_message(msg_id)
                future = asyncio.run_coroutine_threadsafe(coro, self.loop)
                msg = future.result()
            except discord.errors.NotFound:
                msg_id = None
                msg = None

            if msg:
                coro = msg.edit(content=msg_s)
                asyncio.run_coroutine_threadsafe(coro, self.loop)

        if not msg_id:
            msg_id = self._send_message(msg_s)
            self.last_message_ids[ptype] = msg_id
            self.all_message_ids.add(msg_id)

    # --------------------------------

    def start(self):
        self._init_discord()

    def end(self):
        self._quit_discord()

    # --------------------------------

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of the initialization of the :class:`~transformers.Trainer`.
        """
        self._init_discord()

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of training.
        """
        if state.is_local_process_zero:
            self._send_message(f"Begin training on {args.run_name}")

            self.progressbars["train"] = (0, state.max_steps)
            self.timediffs["train"] = time.time()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of training.
        """
        if state.is_local_process_zero:
            self._send_progress_msg("train")
            time_diff = time.time() - self.timediffs["train"]
            self._send_message(
                f"Finish training on {args.run_name}, took: {timedelta(seconds=time_diff)}"
            )

            if "pred" in self.progressbars:
                del self.progressbars["train"]

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of an epoch.
        """
        if state.is_local_process_zero:
            self.timediffs["epoch"] = time.time()
            self.timediffs["step"] = time.time()

            self.last_message_ids.pop("train", None)
            self._send_progress_msg("train")

            msg_id = self._send_message(f"Begin epoch: {state.epoch:.1f}")
            self._delete_later(msg_id, delay=5)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of an epoch.
        """
        if state.is_local_process_zero:
            time_diff = time.time() - self.timediffs["epoch"]
            self._send_message(
                f"Epoch done, took {timedelta(seconds=round(time_diff))!s}"
            )
            self.timediffs["step"] = time.time()

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        # seems to be called at the end of all gradient accumulation sub-steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if state.is_local_process_zero:
            _, max_step = self.progressbars["train"]
            self.progressbars["train"] = (state.global_step, max_step)

            time_diff = time.time() - self.timediffs["step"]
            self._send_progress_msg("train", time_diff=time_diff)
            self.timediffs["step"] = time.time()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after an evaluation phase.
        """
        if state.is_local_process_zero:
            if "pred" in self.progressbars:
                del self.progressbars["pred"]

            msg_id = self._send_message(f"After eval ...")
            self._delete_later(msg_id, delay=5)
            # TODO: reset step timer?

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after a checkpoint save.
        """
        msg_id = self._send_message(f"Saving in epoch {state.epoch:.1f}")
        self._delete_later(msg_id, delay=5)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after logging the last logs.
        """
        # is_world_process_zero
        # is_local_process_zero
        if state.is_local_process_zero:
            logs = kwargs["logs"]
            self._send_log_results(logs, state, args)

    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after a prediction step.
        """
        if state.is_local_process_zero:
            eval_dataloader = kwargs["eval_dataloader"]
            if "pred" not in self.progressbars:
                self.progressbars["pred"] = (0, len(eval_dataloader))
            cur_step, max_step = self.progressbars["pred"]
            self.progressbars["pred"] = (cur_step + 1, max_step)

            if "pred" not in self.timediffs:
                self.timediffs["pred"] = time.time()
            time_diff = time.time() - self.timediffs["pred"]
            self.timediffs["pred"] = time.time()

            self._send_progress_msg("pred", time_diff=time_diff)

    def __del__(self):
        # NOTE: probably never called as there will still be some references
        # left somewhere (thread, callbacks?, self.*?)
        msg_id = self._send_message(f"Ending.")
        self._delete_later(msg_id, delay=3)
        self._quit_discord()


# ----------------------------------------------------------------------------
