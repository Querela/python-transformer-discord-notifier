__version__ = "0.1.0"

import logging
import time
from datetime import timedelta

from tqdm import tqdm
from transformers.trainer import DataLoader
from transformers.trainer_callback import ProgressCallback
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_callback import TrainerControl
from transformers.trainer_callback import TrainerState
from transformers.training_args import TrainingArguments

from typing import Optional, Union, Dict, Any, Tuple

from .discord import DiscordClient


LOGGER = logging.getLogger(__name__)


__all__ = ["DiscordProgressCallback"]


# ----------------------------------------------------------------------------


class MessageWrapperTQDMWriter:
    def __init__(
        self,
        client: DiscordClient,
        msg_fmt: str,
        delete_after: bool = True,
    ):
        self.client = client

        self.msg_id: Optional[int] = None
        self.delete_after = delete_after

        self.msg_fmt = msg_fmt
        self.last_msg: Optional[str] = None

    def write(self, text: str):
        text = text.strip("\r\n")
        if not text.strip():
            return

        self.last_msg = text

        msg_s = self.msg_fmt.format(text=text)
        self.msg_id = self.client.update_or_send_message(msg_id=self.msg_id, text=msg_s)

    def flush(self):
        pass

    def close(self):
        if self.delete_after and self.msg_id is not None and self.client:
            try:
                self.client.delete_later(self.msg_id, delay=10)
            except AttributeError:
                pass
            self.msg_id = None

    def __del__(self):
        LOGGER.info("__del__ of MessageWrapperTQDMWriter")
        self.close()


class DiscordProgressCallback(ProgressCallback):
    def __init__(
        self, token: Optional[str] = None, channel: Optional[Union[str, int]] = None
    ):
        super().__init__()

        self.client = DiscordClient(token, channel)

        self.last_embed_id: Optional[int] = None
        self.epoch_start_time: Optional[float] = None

        self.writer_train: Any = None
        self.writer_predict: Any = None

    # --------------------------------

    def start(self) -> None:
        """Start the Discord bot."""
        self.client.init()

    def end(self) -> None:
        """Stop the Discord bot. Cleans up resources."""
        self.client.quit()

    # --------------------------------

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_init_end(args, state, control, **kwargs)
        self.client.init()

    def __del__(self):
        self.client.quit()

    # --------------------------------

    def _new_tqdm_bar(
        self,
        desc: str,
        msg_fmt: str,
        delete_after: bool = True,
        **kwargs,
    ) -> Tuple[tqdm, MessageWrapperTQDMWriter]:
        """Builds an internal ``tqdm`` wrapper for progress tracking.

        Patches its ``file.write`` method to forward it to Discord.
        Tries to update existing messages to avoid spamming the channel.
        """

        writer = MessageWrapperTQDMWriter(
            self.client,
            msg_fmt=msg_fmt,
            delete_after=delete_after,
        )
        pgbr = tqdm(
            desc=desc,
            ascii=False,
            leave=False,
            position=0,
            file=writer,
            **kwargs,
        )

        return pgbr, writer

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_local_process_zero:
            msg_fmt = "```\n{text}\n```"
            if args.run_name:
                msg_fmt = f"Run: **{args.run_name}**\n{msg_fmt}"

            self.training_bar, self.writer_train = self._new_tqdm_bar(
                desc="train",
                msg_fmt=msg_fmt,
                delete_after=False,
                total=state.max_steps,
            )
        self.current_step = 0

    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        eval_dataloader: Optional[DataLoader] = None,
        **kwargs,
    ):
        if state.is_local_process_zero:
            if self.prediction_bar is None:
                if self.writer_predict is None:
                    msg_fmt = "```\n{text}\n```"
                    if args.run_name:
                        msg_fmt = f"Run: **{args.run_name}**\n{msg_fmt}"
                else:
                    msg_fmt = self.writer_predict.msg_fmt

                self.prediction_bar, self.writer_predict = self._new_tqdm_bar(
                    desc="predict",
                    msg_fmt=msg_fmt,
                    delete_after=True,
                    total=len(eval_dataloader),
                )
            self.prediction_bar.update(1)

    # --------------------------------

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_epoch_begin(args, state, control, **kwargs)
        if state.is_local_process_zero:
            self.epoch_start_time = time.time()

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_epoch_end(args, state, control, **kwargs)
        if state.is_local_process_zero:
            time_diff = time.time() - self.epoch_start_time

            self.writer_train.msg_fmt = self.writer_train.msg_fmt.format(
                text=(
                    f"{self.writer_train.last_msg}\n"
                    f"  Epoch {int(state.epoch)}: "
                    f"{timedelta(seconds=round(time_diff))!s}\n"
                    f"{{text}}"
                )
            )

    # --------------------------------

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_train_end(args, state, control, **kwargs)
        if state.is_local_process_zero:
            if self.writer_train is not None:
                self.writer_train.close()
                self.writer_train = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_evaluate(args, state, control, **kwargs)
        if state.is_local_process_zero:
            self.writer_predict.msg_fmt = self.writer_predict.msg_fmt.format(
                text=f"{self.writer_predict.last_msg}\n{{text}}"
            )
            if self.prediction_bar is not None:
                self.prediction_bar.close()
                self.prediction_bar = None
            if self.writer_predict is not None:
                self.writer_predict.close()

    # --------------------------------

    def _send_log_results(
        self,
        logs: Dict[str, Any],
        state: TrainerState,
        args: TrainingArguments,
        is_train: bool,
    ) -> int:
        """Formats current log metrics as Embed message.

        Given a huggingface transformers Trainer callback parameters,
        we create an :class:`discord.Embed` with the metrics as key-values.
        Send the message and returns the message id."""
        results_embed = self.client.build_embed(
            kvs=logs,
            title="Results (training)" if is_train else "Results (evaluation)",
            footer=f"Global step: {state.global_step} | Run: {args.run_name}",
        )

        return self.client.send_message(text="", embed=results_embed)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if state.is_local_process_zero:
            is_train = False
            if self.training_bar is not None:
                is_train = True
                _ = logs.pop("total_flos", None)
            msg_id = self._send_log_results(logs, state, args, is_train)
            self.last_embed_id = msg_id

    # --------------------------------


# ----------------------------------------------------------------------------


class DiscordFancyCallback(TrainerCallback):
    def __init__(
        self, token: Optional[str] = None, channel: Optional[Union[str, int]] = None
    ):
        super().__init__()

        self.client = DiscordClient(token, channel)

        self.last_message_ids: Dict[str, int] = dict()
        self.progressbars: Dict[str, int] = dict()
        self.timediffs: Dict[str, float] = dict()

    # --------------------------------

    def start(self) -> None:
        """Start the Discord bot."""
        self.client.init()

    def end(self) -> None:
        """Stop the Discord bot. Cleans up resources."""
        self.client.quit()

    # --------------------------------

    def send_log_results(
        self, logs: Dict[str, Any], state: TrainerState, args: TrainingArguments
    ) -> int:
        """Formats current log metrics as Embed message.

        Given a huggingface transformers Trainer callback parameters,
        we create an :class:`discord.Embed` with the metrics as key-values.
        Send the message and returns the message id."""
        results_embed = self.client.build_embed(
            kvs=logs,
            title="Results",
            footer=f"Global step: {state.global_step} | Run: {args.run_name}",
        )

        return self.client.send_message(text="", embed=results_embed)

    def send_progress_msg(self, ptype: str, time_diff: Optional[float] = None) -> None:
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

        new_msg_id = self.client.update_or_send_message(text=msg_s, msg_id=msg_id)
        if new_msg_id != msg_id:
            self.last_message_ids[ptype] = msg_id

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
        self.client.init()

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
            self.client.send_message(f"Begin training on {args.run_name}")

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
            self.send_progress_msg("train")
            time_diff = time.time() - self.timediffs["train"]
            self.client.send_message(
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
            self.send_progress_msg("train")

            msg_id = self.client.send_message(f"Begin epoch: {state.epoch:.1f}")
            self.client.delete_later(msg_id, delay=5)

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
            self.client.send_message(
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
            self.send_progress_msg("train", time_diff=time_diff)
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

            msg_id = self.client.send_message("After eval ...")
            self.client.delete_later(msg_id, delay=5)
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
        msg_id = self.client.send_message(f"Saving in epoch {state.epoch:.1f}")
        self.client.delete_later(msg_id, delay=5)

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
            self.send_log_results(logs, state, args)

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

            self.send_progress_msg("pred", time_diff=time_diff)

    def __del__(self):
        # NOTE: probably never called as there will still be some references
        # left somewhere (thread, callbacks?, self.*?)
        # NOTE: would need to time.sleep(..) to allow async tasks to finish before stopping loop
        # msg_id = self.client.send_message("Ending.")
        # self.client.delete_later(msg_id, delay=3)
        self.client.quit()


# ----------------------------------------------------------------------------
