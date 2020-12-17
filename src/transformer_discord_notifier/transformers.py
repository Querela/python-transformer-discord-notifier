import logging
import time
from concurrent.futures import TimeoutError
from datetime import timedelta

from tqdm import tqdm
from transformers.trainer_callback import ProgressCallback
from transformers.trainer_callback import TrainerControl
from transformers.trainer_callback import TrainerState
from transformers.training_args import TrainingArguments

# from transformers.trainer import DataLoader

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
        LOGGER.debug("__del__ of MessageWrapperTQDMWriter")
        self.close()


class DiscordProgressCallback(ProgressCallback):
    """An extended :class:`transformers.trainer_callback.ProgressCallback`
    that logs training and evaluation progress and statistics to a Discord
    channel.

    Attributes
    ----------
    client : DiscordClient
        a blocking Discord client
    disabled : bool
        ``True`` if Discord client couldn't not be initialized
        successfully, all callback methods are disabled silently
    """

    def __init__(
        self, token: Optional[str] = None, channel: Optional[Union[str, int]] = None
    ):
        """
        Parameters
        ----------
        token : Optional[str], optional
            Discord bot token, by default None
        channel : Optional[Union[str, int]], optional
            Discord channel name or numeric id, by default None
        """
        super().__init__()

        self.disabled = True
        self.client = DiscordClient(token, channel)

        self.last_embed_id: Optional[int] = None
        self.epoch_start_time: Optional[float] = None

        self.writer_train: Any = None
        self.writer_predict: Any = None

    # --------------------------------

    def start(self) -> None:
        """Start the Discord bot."""
        is_ok, err_msg = True, None
        try:
            self.client.init()
            self.disabled = False
        except (RuntimeError, TimeoutError, TypeError) as ex:
            is_ok = False
            err_msg = str(ex)

        if not is_ok or not self.client or not self.client._initialized:
            LOGGER.warning(
                "Failure to initialize Discord client."
                " Silently disable callback handler."
                + (f" Error: {err_msg}" if err_msg else "")
            )
            self.disabled = True

    def end(self) -> None:
        """Stop the Discord bot. Cleans up resources."""
        if self.client:
            self.client.quit()
        self.disabled = True

    # --------------------------------

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.start()

    def __del__(self):
        self.end()

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
        if self.disabled:
            return

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
        eval_dataloader=None,
        **kwargs,
    ):
        if self.disabled:
            return

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

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.disabled:
            return

        super().on_step_end(args, state, control, **kwargs)

    # --------------------------------

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.disabled:
            return

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
        if self.disabled:
            return

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
        if self.disabled:
            return

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
        if self.disabled:
            return

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
        if self.disabled:
            return

        if state.is_local_process_zero:
            is_train = False
            if self.training_bar is not None:
                is_train = True
                _ = logs.pop("total_flos", None)
            msg_id = self._send_log_results(logs, state, args, is_train)
            self.last_embed_id = msg_id

    # --------------------------------


# ----------------------------------------------------------------------------
