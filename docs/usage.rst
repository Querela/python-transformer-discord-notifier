=====
Usage
=====

Using ``DiscordProgressCallback``
---------------------------------

How to use the :class:`~transformer_discord_notifier.transformers.DiscordProgressCallback` in a huggingface.co Transformer in a project/training script:

.. code-block:: python
	:linenos:
	:emphasize-lines: 3,9-10,20,26

	from transformers import Trainer
	# ... other import ...
	from transformer_discord_notifier import DiscordProgressCallback

	def run_trainer():
		# ... set up things beforehand ...

		# Initialize the Discord bot
		dpc = DiscordProgressCallback(token=None, channel=None, create_experiment_channels=False)
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

Alternatively, since version `v0.2.0` it is possible to omit the starting and stopping of the :class:`~transformer_discord_notifier.transformers.DiscordProgressCallback`, and it can be used like any other `huggingface.co callback handler <https://huggingface.co/transformers/main_classes/callback.html>`_:

.. code-block:: python
	:linenos:
	:emphasize-lines: 3,16

	from transformers import Trainer
	# ... other import ...
	from transformer_discord_notifier import DiscordProgressCallback

	def run_trainer():
		# ... set up transformer stuff beforehand ...

		# Initialize our Trainer
		trainer = Trainer(
			model=model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=eval_dataset,
			# ...
			# add our callback to the trainer
			callbacks=[DiscordProgressCallback]
		)

		# ... do things like train/eval/predict
		# ... when the trainer instance is garbage collected, it will clean up the Discord bot

Note, however, that the both ``token`` and ``channel`` should be provided, either as class initialization parameters or as environment variables, ``DISCORD_TOKEN`` and ``DISCORD_CHANNEL``. The handler will try to load from environment variables if the instance properties are ``None``. Both should be explicitely provided to have it working correctly!

Since version `v0.5.0`, we included the ability to create separate experiment channels.
To enable those, set the following environment variables:

.. code-block:: bash
	
	# yes|y|true|t|1 will enable the channel creation
	export DISCORD_CREATE_EXPERIMENT_CHANNEL=yes
	
	# (optional) set (or create) a category channel for the experiment channels
	export DISCORD_EXPERIMENT_CATEGORY="All my Experiments"
	# (optional) override and set the run_name / experiment name
	# new text channel name, note, that it will be all lowercase, "-" for whitespaces
	export DISCORD_EXPERIMENT_NAME="Experiment-Run-A1B2"

How to setup a Discord bot
--------------------------

How to setup a Discord bot, how to get the token or the channel id? Please visit the following links:

- `How to create a bot? <https://discordpy.readthedocs.io/en/latest/discord.html>`_
- Related project `discord-notifier-bot <https://github.com/Querela/discord-notifier-bot#bot-creation-etc>`_, setup guide in README
