import asyncio
import logging
import os
import threading
from pprint import pformat

import discord

from typing import Optional, Union, Dict, Set, Any


LOGGER = logging.getLogger(__name__)


__all__ = ["DiscordClient"]


# ----------------------------------------------------------------------------


class MyClient(discord.Client):
    async def on_ready(self):
        LOGGER.info("Logged on as {0}!".format(self.user))

    async def on_message(self, message):
        LOGGER.debug("Message from {0.author}: {0.content}".format(message))


# ----------------------------------------------------------------------------


class DiscordClient:
    """A blocking wrapper around the asyncio Discord.py client."""

    def __init__(
        self, token: Optional[str] = None, channel: Optional[Union[str, int]] = None
    ):
        self._discord_token: Optional[str] = token
        self._discord_channel: Optional[Union[str, int]] = channel

        self.all_message_ids: Set[int] = set()

        self._initialized: bool = False
        self.client_thread: Optional[threading.Thread] = None
        self.client: Optional[discord.Client] = None

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    # --------------------------------

    def _load_credentials(self) -> None:
        """Try to load missing Discord configs (token, channel) from
        environment variables."""
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
        """Try to find a writable text channel.

        Follow the following algorithm:

            1. if ``name`` is being provided, search for this channel first
            2. if not found, search for ``self._discord_channel``, then
               channel that can be configured on instance creation or by
               loading environment variables. Check first for a channel with
               the given name as string, then fall back to an integer
               channel id.
            3. if still not found, search for a channel with a given default
               name, like "default" or "Allgemein". As this seems to depend
               on the language, it might not find one.

        If after all this still no channel has been found, either because no
        channel with the given names/id exists, or because the Discord token
        gives no acces to guilds/channels which we have access to, we throw
        a ``RuntimeError``. We now can't use this callback handler.

        Parameters
        ----------
        name : Optional[str], optional
            channel name to search for first, by default None
        default_name : str, optional
            alternative default Discord channel name, by default "default"

        Returns
        -------
        int
            channel id

        Raises
        ------
        RuntimeError
            raised if no `guild` Discord server found (i.e. Discord bot
            has no permissions / was not yet invited to a Discord server)
        RuntimeError
            raised if channel could not be found
        """
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

    def init(self) -> None:
        """Initialize Discord bot for accessing Discord/writing messages.

        It loads the credentials, starts the asyncio Discord bot in a
        separate thread and after connecting searches for our target channel.
        """
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

            async def client_runner():
                try:
                    await self.client.start(self._discord_token)
                except asyncio.CancelledError:
                    pass
                except Exception as ex:
                    LOGGER.debug("client_runner: Error? %s", ex)
                finally:
                    if not self.client:
                        # just to be sure, should never happen
                        return

                    LOGGER.debug(
                        "client_runner: close() - is_closed: %s",
                        self.client.is_closed(),
                    )
                    if not self.client.is_closed():
                        await self.client.close()

                LOGGER.debug("client_runner: done.")

            def stop_loop_on_completion(_future):
                if self.loop and not self.loop.is_closed:
                    self.loop.stop()

            future = asyncio.ensure_future(client_runner(), loop=self.loop)
            future.add_done_callback(stop_loop_on_completion)

            try:
                self.loop.run_forever()
            finally:
                LOGGER.debug("Closing Discord AsyncIO Loop ...")
                future.remove_done_callback(stop_loop_on_completion)
                self.loop.close()
                LOGGER.debug("Discord AsyncIO Loop closed.")

        self.client_thread = threading.Thread(
            target=client_thread_func, name="discord-asyncio", daemon=True
        )
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
        future = asyncio.run_coroutine_threadsafe(
            self.client.wait_until_ready(), self.loop
        )
        _ = future.result()

        search_text_channel()

        LOGGER.debug("Discord handler initialized.")

    def quit(self) -> None:
        """Shutdown the Discord bot.

        Tries to close the Discord bot safely, closes the asyncio loop,
        waits for the background thread to stop (deamonized, so on program
        exit it will quit anyway)."""
        if not self.client:
            LOGGER.debug("Discord already shutdown, do nothing.")
            return

        LOGGER.debug("Shutdown Discord handler ...")

        def _cancel_tasks(loop):
            """Cancel reamining tasks. Try to wait until finished."""
            # Code adapted from discord.client to work with threads
            try:
                task_retriever = asyncio.Task.all_tasks
            except AttributeError:
                # future proofing for 3.9 I guess
                task_retriever = asyncio.all_tasks

            tasks = {t for t in task_retriever(loop=loop) if not t.done()}

            if not tasks:
                return

            LOGGER.info("Cleaning up after %d tasks.", len(tasks))
            for task in tasks:
                task.cancel()

            LOGGER.info("All tasks finished cancelling.")

            future = asyncio.gather(*tasks, loop=loop, return_exceptions=True)
            coro = asyncio.wait_for(future, timeout=5, loop=loop)
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            future.result()

            for task in tasks:
                if task.cancelled():
                    continue
                try:
                    if task.exception() is not None:
                        loop.call_exception_handler(
                            {
                                "message": "Unhandled exception during Client.run shutdown.",
                                "exception": task.exception(),
                                "task": task,
                            }
                        )
                except Exception as ex:
                    LOGGER.debug("task cancel error? %s %s", task, ex)
                    continue

        # stop client
        coro = self.client.close()
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        future.result(timeout=10)

        # cancel remaining tasks
        self.loop.call_soon_threadsafe(_cancel_tasks, self.loop)

        # stop loop
        self.loop.stop()

        # asyncio background thread should have finished, but wait
        # if still not joined after timeout, just quit
        # (thread will stop on program end)
        self.client_thread.join(timeout=3)

        self.client = None
        self.client_thread = None
        self.loop = None

    # --------------------------------

    def send_message(
        self, text: str = "", embed: Optional[discord.Embed] = None
    ) -> Optional[int]:
        """Sends a message to our Discord channel. Returns the message id.

        Parameters
        ----------
        text : str, optional
            text message to send, by default ""
        embed : Optional[discord.Embed], optional
            embed object to attach to message, by default None

        Returns
        -------
        Optional[int]
            message id if `text` and `embed` were both not ``None``,
            ``None`` if nothing was sent
        """
        # if nothing to send, return
        if not text and not embed:
            return None

        async def _send():
            await self.client.wait_until_ready()

            channel: discord.TextChannel = self.client.get_channel(
                self._discord_channel
            )
            msg: discord.Message = await channel.send(text, embed=embed)
            return msg

        future = asyncio.run_coroutine_threadsafe(_send(), self.loop)
        message = future.result()
        self.all_message_ids.add(message.id)
        return message.id

    def update_or_send_message(
        self,
        text: str = "",
        embed: Optional[discord.Embed] = None,
        msg_id: Optional[int] = None,
    ) -> Optional[int]:
        """Wrapper for :func:`send_message` to updated an existing message,
        identified by `msg_id` or simply send a new message if no prior
        message found.

        Parameters
        ----------
        text : str, optional
            text message, by default ""
        embed : Optional[discord.Embed], optional
            Discord embed, by default None
        msg_id : Optional[int], optional
            message id of prior message sent in channel, if not provided
            then send a new message.

        Returns
        -------
        Optional[int]
            message id of updated or newly sent message,
            ``None`` if nothing was sent
        """
        if msg_id:
            try:
                channel: discord.TextChannel = self.client.get_channel(
                    self._discord_channel
                )
                coro = channel.fetch_message(msg_id)
                future = asyncio.run_coroutine_threadsafe(coro, self.loop)
                msg: discord.Message = future.result()
            except discord.errors.NotFound:
                msg_id = None
                msg = None

            if msg:
                coro = msg.edit(content=text, embed=embed)
                asyncio.run_coroutine_threadsafe(coro, self.loop)

        if not msg_id:
            msg_id = self.send_message(text=text, embed=embed)

        return msg_id

    def delete_later(self, msg_id: int, delay: Union[int, float] = 5) -> bool:
        """Runs a delayed message deletion function.

        Parameters
        ----------
        msg_id : int
            message id of message sent in Discord channel
        delay : Union[int, float], optional
            delay in seconds for then to delete the message, by default 5

        Returns
        -------
        bool
            ``True`` if message deletion is queued,
            ``False`` if message could not be found in channel
        """
        try:
            channel: discord.TextChannel = self.client.get_channel(
                self._discord_channel
            )
            coro = channel.fetch_message(msg_id)
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            message: discord.Message = future.result()
        except discord.errors.NotFound:
            return False

        coro = message.delete(delay=delay)
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)

        return True

    @staticmethod
    def build_embed(
        kvs: Dict[str, Any],
        title: Optional[str] = None,
        footer: Optional[str] = None,
    ) -> discord.Embed:
        """Builds an Embed message from key-values.

        Parameters
        ----------
        kvs : Dict[str, Any]
            Key-Value dictionary for embed fields, non ``int``/``float``
            values will be formatted with :func:`pprint.pformat`
        title : Optional[str], optional
            title string, by default None
        footer : Optional[str], optional
            footer string, by default None

        Returns
        -------
        discord.Embed
            embed object to send via :meth:`send_message`
        """
        embed = discord.Embed(title=title)

        for k, v in kvs.items():
            if isinstance(v, (int, float)):
                embed.add_field(name=k, value=v, inline=True)
            else:
                embed.add_field(
                    name=k, value=f"```json\n{pformat(v)}\n```", inline=False
                )

        if footer:
            embed.set_footer(text=footer)
        return embed

    # --------------------------------


# ----------------------------------------------------------------------------
