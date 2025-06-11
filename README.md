# DEM-evaluation

This repository was built to streamline the comparison of different DSM pipelines. <br>
The Dockerfile can be used to install several pipelines, such as Nasa's ASP, CNES' CARS, Centre Borelli's s2p-hd, IGN's Micmac.

The functions in `comparison.py` can be used to compute statistics for the DSM reconstructions of each method. The code used to compare DSMs to ground truths is dicretly inspired from the library [demcompare](https://github.com/CNES/demcompare.git) which we thank. 

## References

If you use this software please cite the following paper:

[*s2p-hd: Gpu-Accelerated Binocular Stereo Pipeline for Large-Scale 
Same-Date Stereo*](https://hal.science/view/index/docid/5051235), Tristan Amadei, 
Enric Meinhardt-Llopis, Carlo de Franchis, Jeremy Anger, Thibaud Ehret, 
Gabriele Facciolo. CVPR EarthVision 2025.