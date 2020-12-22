========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |gha-tox| |travis|
        | |coveralls|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/python-transformer-discord-notifier/badge/?style=flat
    :target: https://readthedocs.org/projects/python-transformer-discord-notifier
    :alt: Documentation Status

.. |gha-tox| image:: https://github.com/Querela/python-transformer-discord-notifier/workflows/Python%20tox%20tests/badge.svg
    :alt: Python tox tests
    :target: https://github.com/Querela/python-transformer-discord-notifier/actions?query=workflow%3A%22Python+tox+tests%22

.. |travis| image:: https://api.travis-ci.org/Querela/python-transformer-discord-notifier.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/Querela/python-transformer-discord-notifier

.. |coveralls| image:: https://coveralls.io/repos/Querela/python-transformer-discord-notifier/badge.svg?branch=master&service=github
    :alt: Coverage Status
    :target: https://coveralls.io/r/Querela/python-transformer-discord-notifier

.. |version| image:: https://img.shields.io/pypi/v/transformer-discord-notifier.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/transformer-discord-notifier

.. |wheel| image:: https://img.shields.io/pypi/wheel/transformer-discord-notifier.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/transformer-discord-notifier

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/transformer-discord-notifier.svg
    :alt: Supported versions
    :target: https://pypi.org/project/transformer-discord-notifier

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/transformer-discord-notifier.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/transformer-discord-notifier

.. |commits-since| image:: https://img.shields.io/github/commits-since/Querela/python-transformer-discord-notifier/v0.4.4.svg
    :alt: Commits since latest release
    :target: https://github.com/Querela/python-transformer-discord-notifier/compare/v0.4.4...master



.. end-badges

A Discord Notifier to send progress updates, params and results to a Discord channel.

* Free software: MIT license

Installation
============

::

    pip install transformer-discord-notifier

You can also install the in-development version with::

    pip install https://github.com/Querela/python-transformer-discord-notifier/archive/master.zip


Documentation
=============


https://python-transformer-discord-notifier.readthedocs.io/

::

    git clone https://github.com/Querela/python-transformer-discord-notifier.git
    cd python-transformer-discord-notifier
    sphinx-build -b html docs dist/docs


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox