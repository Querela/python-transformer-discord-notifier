=====
Usage
=====

To use Transformer Discord Notifier in a project::

	from transformers import Trainer
	# ... other import ...
	from transformer_discord_notifier import DiscordProgressCallback

	def run_trainer():
		# ... set up things beforehand ...

		# Initialize the Discord bot
		dpc = DiscordProgressCallback(token=None, channel=None)
		dpc.start()

		# Initialize our Trainer
		trainer = Trainer(
			model=model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=eval_dataset,
			# ...
			# add our callback to the trainer
			callbacks=[dpc]
		)

		# ... do things like train/eval/predict

		# shutdown our discord handler as it would continue to run indefinitely
		dpc.end()

Note, however, that the both ``token`` and ``channel`` should be provided, either as class initialization parameters or as environment variables, ``DISCORD_TOKEN`` and ``DISCORD_CHANNEL``. The handler will try to load from environment variables if the instance properties are ``None``. Both should be explicitely provided to have it working correctly!

How to setup a Discord bot (How to get the Token?) or the channel id? Please visit the following links:

- `How to create a bot? <https://discordpy.readthedocs.io/en/latest/discord.html>`_
- Related project `discord-notifier-bot <https://github.com/Querela/discord-notifier-bot#bot-creation-etc>`_, setup guide in README
