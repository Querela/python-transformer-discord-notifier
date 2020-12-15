transformer_discord_notifier.discord
====================================

.. automodule:: transformer_discord_notifier.discord
    :members:
    :private-members:

The :class:`~transformer_discord_notifier.discord.DiscordClient` can be used standalone, but it might be easier to just extract the module code to avoid having to install all the related `transformers <https://huggingface.co/transformers/>`_ requirements. It wraps the asyncio `Discord.py client <https://discordpy.readthedocs.io/>`_ inside a background thread and makes its calls essentially blocking. This eases the usage of it in foreign code that does not uses `asyncio <https://docs.python.org/3/library/asyncio.html>`_.

.. code-block:: python
    :linenos:

    from transformer_discord_notifier.discord import DiscordClient

    # configuration
    token = "abc123.xyz..."
    channel = "Allgemein"

    # create client and start background thread, to connect/login ...
    # if token/channel are None, it will try to load from environment variables
    client = DiscordClient(token=token, channel=channel)
    client.init()

    # send message
    msg_id = client.send_message("test")

    # update message content
    msg_id = client.update_or_send_message(text="abc", msg_id=msg_id)

    # delete it after 3.1 seconds,
    # NOTE: this call will not block!
    client.delete_later(msg_id, delay=3.1)

    # quit client (cancel outstanding tasks!, quit asyncio thread)
    client.quit()
