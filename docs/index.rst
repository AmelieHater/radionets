:html_theme.sidebar_secondary.remove: true
:html_theme.sidebar_primary.remove: true

.. _radionets:

.. show title in tab name but not on index page
.. raw:: html

   <div style="height: 0; visibility: hidden;">

=========
Radionets
=========

.. raw:: html

   </div>

.. currentmodule:: radionets

.. image:: ../assets/radionets_large_subtitle.png
   :class: only-light
   :align: center
   :width: 90%
   :alt: The radionets logo.

.. image:: ../assets/radionets_large_subtitle_dark.png
   :class: only-dark
   :align: center
   :width: 90%
   :alt: The radionets logo.

|

**Version**: |version| | **Date**: |today|

**Useful links**:
`Source Repository <https://github.com/radionets-project/radionets>`__ |
`Issue Tracker <https://github.com/radionets-project/radionets/issues>`__ |
`Pull Requests <https://github.com/radionets-project/radionets/pulls>`__

**License**: `MIT <https://github.com/radionets-project/radionets/blob/main/LICENSE>`__

**Python**: |python_requires|

``radionets`` is a deep-learning framework for the simulation and analysis of radio interferometric data in Python.
The goal is to reconstruct calibrated observations with convolutional Neural Networks to create
high-resolution images. For further information, please have a look at our
`paper <https://www.aanda.org/component/article?access=doi&doi=10.1051/0004-6361/202142113>`__.


Analysis strategies leading to reproducible processing and evaluation of data recorded by radio interferometers
include:

- Simulation of datasets (see also the https://github.com/radionets-project/radiosim repository)
- Simulation of radio interferometer observations (see also the https://github.com/radionets-project/pyvisgen repository)
- Training of deep learning models
- Reconstruction of radio interferometric data

.. _radionets_docs:

.. toctree::
  :maxdepth: 1
  :hidden:

  user-guide/index
  developer-guide/index
  api-reference/index
  changelog
  citeus
  references
  glossary



.. grid:: 1 2 2 3

    .. grid-item-card::
        :class-item: animated-sd-card

        :octicon:`book;40px`

        User Guide
        ^^^^^^^^^^

        Learn how to get started as a user. This guide
        will help you install radionets.

        +++

        .. button-ref:: user-guide/index
            :expand:
            :color: primary
            :click-parent:

            To the user guide


    .. grid-item-card::
        :class-item: animated-sd-card

        :octicon:`person-add;40px`

        Developer Guide
        ^^^^^^^^^^^^^^^

        Learn how to get started as a developer.
        This guide will help you install radionets for development
        and explains how to contribute.

        +++

        .. button-ref:: developer-guide/index
            :expand:
            :color: primary
            :click-parent:

            To the developer guide


    .. grid-item-card::
        :class-item: animated-sd-card

        :octicon:`code;40px`

        API Docs
        ^^^^^^^^

        The API docs contain detailed descriptions of
        of the various modules, classes and functions
        included in radionets.

        +++

        .. button-ref:: api-reference/index
            :expand:
            :color: primary
            :click-parent:

            To the API docs
