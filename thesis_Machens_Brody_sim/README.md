To re-create the simulation, you need the original Matlab code used by Machens Romo and Brody.

First, download the original code from http://science.sciencemag.org/content/307/5712/1121, the exact link is http://science.sciencemag.org/highwire/filestream/586751/field_highwire_adjunct_files/0/Machens.Matlabcode.SOM.zip .

Then extract the zipped file within this folder (that should create a "twonode" folder), and compile the "mex" files within that, as instructed by the original code's README.

Finally running "thesis_create_experimentData_spiking" should recreate the simulated spiking data.
