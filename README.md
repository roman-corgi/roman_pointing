# Roman Pointing

This repository includes basic utilities to compute Roman observatory pointing angles, specifically the sun angle (angle between the unit vector pointing from the observatory to the target and the unit vector pointing from the observatory and the sun), along with the pitch and yaw settings such that the observatory boresight points at the target. 

The observatory orientation zero-point is such that the pitch angle will be the same as values computed for OS11.  The yaw, angle, however, will be different.  If the observatory is placed exactly at L2, then the yaw will be equal to the OS11 value plus 180 degrees.

The Jupyter notebook in the `Notebooks` folder demonstrates how to use these utilities. 

To install the backend, clone or download this repository, navigate to the top-level directory of the repository (the one containing file `pyproject.toml`) and run:

```
pip install .
```


# Roman Pointing Interface

Use notebook ``Roman Space Telescope Keepout Map Generator.ipynb`` for offline execution (internet connnection still required).

To run in Colab, first go to:
https://colab.research.google.com/github/roman-corgi/roman_pointing/blob/main/Notebooks/00_Google_Colab_Setup.ipynb

Ensure that you are logged in with the Google account you wish to use (data will be written to the Google Drive associated with this account). You can check which account you are logged into by clicking on the user icon in the top right-hand corner of the page.

Execute all of the cells in the notebook, responding to any pop-up prompts along the way (see the notebook for more detailed instructions). Note that you only need to run this notebook **once** (even if you log out/close the browser instance, the files written to your Google Drive will be persistent).

After successfully executing the setup notebook, go to:
https://colab.research.google.com/github/roman-corgi/roman_pointing/blob/main/Notebooks/01%20-%20Colab%20Roman%20Space%20Telescope%20Keepout%20Map%20Generato.ipynb

Click the 'Run all' button at the top of the screen.

For any subsequent sessions, go directly to this link. 
