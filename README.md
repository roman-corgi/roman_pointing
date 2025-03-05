# Roman Pointing

This repository includes basic utilities to compute Roman observatory pointing angles, specifically the sun angle (angle between the unit vector pointing from the observatory to the target and the unit vector pointing from the observatory and the sun), along with the pitch and yaw settings such that the observatory boresight points at the target. 

The observatory orientation zero-point is such that the pitch angle will be the same as values computed for OS11.  The yaw, angle, however, will be different.  If the observatory is placed exactly at L2, then the yaw will be equal to the OS11 value plus 180 degrees.

The Jupyter notebook in the `Notebooks` folder demonstrates how to use these utilities. 
