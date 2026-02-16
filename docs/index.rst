.. roman_pointing documentation master file, created by
   sphinx-quickstart on Mon Feb 16 09:20:07 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

roman_pointing documentation
============================

`roman_pointing` includes basic utilities to compute Roman observatory pointing angles, specifically the sun angle (angle between the unit vector pointing from the observatory to the target and the unit vector pointing from the observatory and the sun), along with the pitch and yaw settings such that the observatory boresight points at the target. 

The observatory orientation zero-point is such that the pitch angle will be the same as values computed for OS11.  The yaw, angle, however, will be different.  If the observatory is placed exactly at L2, then the yaw will be equal to the OS11 value plus 180 degrees.

The Jupyter notebook in the ``Notebooks`` folder demonstrates how to use these utilities. 

To install the backend, clone or download the GitHub repository, navigate to the top-level directory of the repository (the one containing file ``pyproject.toml``) and run: ::

    pip install .


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   conventions
   Notebooks/Roman_Pointing
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

