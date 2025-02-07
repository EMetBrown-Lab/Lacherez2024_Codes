# Enhanced diffusion over a periodic trap by hydrodynamic coupling to an elastic mode
## Source codes

Hello !

The files in this repository are the packages you'll need to install in order to run the simulations of our Toy Model.
Everything is commented, but feel free to shoot us an email in case of doubt or issues.

To start, please download your copy of the repo and run the setup.py file.

Then, the automate_simulation.py file can be run. You can choose the parameters you wish to explore in the parameter section.
Depending on the range of parameters, you may want to reduce the time step and simulation time; I left typical values for practicity. 

The automate_simulatiom.py file then runs the run_simulation.py file calling all cython functions of the tm_sinusoid_trap packages, for each combination of parameters you asked for.

All logs are kept in a log folder.

All paramenters are kept in a parameters folder.

Happy coding,

Best,

Juliette
juliette.lacherez@u-bordeaux.fr
