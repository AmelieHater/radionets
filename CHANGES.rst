Radionets 0.3.0 (2023-08-04)
============================


API Changes
-----------


Bug Fixes
---------

- Fixed loading of correct sampling file [`#145 <https://github.com/radionets-project/radionets/pull/145>`__]

- Calculated normalization only on non-zero pixels

  - Fixed typo in rescaling operation [`#149 <https://github.com/radionets-project/radionets/pull/149>`__]

- Fixed sampling for images displayed in real and imaginary part [`#152 <https://github.com/radionets-project/radionets/pull/152>`__]


New Features
------------

- Enabled training and evaluation of half sized images (for 128 pixel images) [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]

- Added naming to save path, so that the files do not get overwritten as easily [`#144 <https://github.com/radionets-project/radionets/pull/144>`__]

- Added normalization callback with two different techniques

  - Updated plotting routines for real/imag images
  - Updated ``evaluate_area`` and ``evaluate_ms_ssim`` for half images
  - Added ``evaluate_ms_ssim`` for sampled images [`#146 <https://github.com/radionets-project/radionets/pull/146>`__]

- Add evaluation of intensity via peak flux and integrated flux comparison [`#150 <https://github.com/radionets-project/radionets/pull/150>`__]

- Centered bin on 1 for histogram evaluation plots

  - Added color to legend [`#151 <https://github.com/radionets-project/radionets/pull/151>`__]

- Added prettier labels and descriptions to plots [`#152 <https://github.com/radionets-project/radionets/pull/152>`__]


Maintenance
-----------

- Deleted unusable functions for new source types
- Deleted unused hardcoded scaling [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]

- Added masked loss functions
- Sorted bundles in simulations
- Minor adjustments in plotting scripts [`#141 <https://github.com/radionets-project/radionets/pull/141>`__]

- Consistent use of batch_size [`#142 <https://github.com/radionets-project/radionets/pull/142>`__]

- Added the model name to predictions and sampling file

  - Deleted unnecessary pad_unsqueeze function
  - Added amp_phase keyword to sample_images
  - Fixed deprecation warning in sampling.py
  - Added image size to test_evaluation.py routines [`#146 <https://github.com/radionets-project/radionets/pull/146>`__]

- Outsourced preprocessing steps in ``train_inspection.py`` [`#148 <https://github.com/radionets-project/radionets/pull/148>`__]

- Removed unused ``norm_path`` from all instances [`#153 <https://github.com/radionets-project/radionets/pull/153>`__]

- Deleted cropping

  - Updated colorbar label
  - Removed ``source_list`` argument [`#154 <https://github.com/radionets-project/radionets/pull/154>`__]


Refactoring and Optimization
----------------------------

- Optimized ``evaluation.utils.trunc_rvs`` with numba, providing functions compiled for cpu and parallel cpu computation. [`#143 <https://github.com/radionets-project/radionets/pull/143>`__]


Radionets 0.2.0 (2023-01-31)
============================


API Changes
-----------

- Train on half-sized iamges and applying symmetry afterward is a backward incompatible change
- Models trained with early versions of ``radionets`` are not supported anymore [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]


Bug Fixes
---------

- Fixed sampling of test data set
- Fixed same indices for plots [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]


New Features
------------

- Enabled training and evaluation of half sized images (for 128 pixel images) [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]


Maintenance
-----------

- Deleted unusable functions for new source types
- Deleted unused hardcoded scaling [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]


Refactoring and Optimization
----------------------------


Radionets 0.1.18 (2023-01-30)
=============================


API Changes
-----------


Bug Fixes
---------


New Features
------------

- Added creation of uncertainty plots
- Changed creation and saving/reading of predictions to ``dicts``

  - Prediction ``dicts`` have 3 or 4 entries depending on uncertainty

- Added scaled option to ``get_ifft``
- Created new dataset class for sampled images
- Created option for sampling and saving the whole test dataset
- Updated and wrote new tests [`#129 <https://github.com/radionets-project/radionets/pull/129>`__]


Maintenance
-----------

- Added and enabled ``towncrier`` in CI. [`#130 <https://github.com/radionets-project/radionets/pull/130>`__]

- Published ``radionets`` on pypi [`#134 <https://github.com/radionets-project/radionets/pull/134>`__]

- Updated README, used figures from the paper, minor text adjustments [`#136 <https://github.com/radionets-project/radionets/pull/136>`__]


Refactoring and Optimization
----------------------------
