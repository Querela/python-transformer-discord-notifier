
Changelog
=========

0.3.1 (2020-12-17)
------------------

* Let Discord bot gracefully handle initialization failures.
* Let transformer callback handler handle invalid configs gracefully, to simply exit.
* Better handling of edge cases of Discord client login.

0.3.0 (2020-12-16)
------------------

* Add (private) scripts (make venv, run checks).
* Update usage docs.
* Extend / rewrite discord client methods.
* Reuse existing ``tqdm`` :class:`transformers.trainer_callback.ProgressCallback` for progress tracking.
* Fancy aggregation of prediction runs, split train progress into epochs.

0.2.1 (2020-12-15)
------------------

* Correct ``setup.py`` validation.
* Add (private) distribution/docs build scripts.

0.2.0 (2020-12-15)
------------------

* Refactor blocking discord code into ``discord`` submodule.
* Fix behaviour for ``__del__`` with refactoring, so it work as intended.
* Improve documentation for ``discord`` module.

0.1.0 (2020-12-11)
------------------

* First release on PyPI.
* First working version, tested manually.
* Cleaned up skeleton files.
* Updated docs.

0.0.0 (2020-12-10)
------------------

* Initial code skeleton.
