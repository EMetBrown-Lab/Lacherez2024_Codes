# Enhanced diffusion over a periodic trap by hydrodynamic coupling to an elastic mode
## Source codes

Hello !

The files in this repository are the packages you'll need to install and/or run to simulate our Toy Model.
Everything is commented, but feel free to shoot us an email in case of doubt or issues.

To start, please download your copy of the repo and run the setup.py file. You'll need Cython installed.

Then, the automate_simulation.py file can be run. You can choose the parameters you wish to explore in the simulation_config.yaml.
Depending on the range of parameters, you may need to modify the sigma in simulation.py, and sometimes the timestep is finnicky: if you choose very high compliances, the automatic time step might be too large to allow for correct convergence, hence my choice to sometimes stick to 1e-3.
I didn't automate it because I didn't need to.

The automate_simulation.py file then runs the run_simulation.py file calling all cython functions of the tm_sinusoid_trap packages, for each combination of parameters you asked for.

All logs are kept in a log folder. Change the naming, maybe, I didnt really need to.

All paramenters are kept in a parameters folder.

Happy coding,

Best,

Juliette
juliette.lacherez@u-bordeaux.fr
