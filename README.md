# Fourier Glare
Create point-spread functions from images of apertures suitable for use in image convolution to simulate glare.

This based on the work by [T. Ritschel, et. al. and their 2009 paper "Temporal Glare: Real-Time Dynamic Simulation of the Scattering in the Human Eye"](https://resources.mpi-inf.mpg.de/hdr/temporalglare/). It is not intended to be realtime, but rather used as part of a compositing pipeline (and, specifically, used in Blender's Convolve node).

## Requirements

* Python 3
* PIP 3
* OpenEXR
* opencv-python
* imageio

## Usage

```python -m fourier-glare -a <file>```

There are other flags supported (or intended to be supported), but that's the main way to do it. There are some sample aperture images you can play with in the `apertures` directory, but feel free to make your own!

Generate files will appear in a directory called `tmp` that, by default, lives inside the module directory. You can override this with a command line switch as desired.

The output result that can be used as a PSF for the Convolve node in the Blender Compositor will be called `chromatic_psf.exr` or `PSF_normalized.exr`, depending on your preference. If you use the former, you will likely need to dial the intensity way down, but you may lose some detail in using `PSF_normalized.exr`.

## Future
I will continue refining this over time. Its original implementation was a full-image convolution, which has been rendered unnecessary by the addition of the Convolve node to Blender's compositor. A lot of vestigial elements remain from that implementation that need to be cleaned up. I'm sure there are optimizations to be done as well.
