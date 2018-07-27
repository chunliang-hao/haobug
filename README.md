# haobug

This is project create to recognize facial emotion through camera.

The author of this project is Chunliang Hao and Jie Shen

# Dependency

You will need the following projects and models to run this project:

1. dlib_and_chehra_stuff 

A repository of code that performs facial landmark tracking, eye point localisation, and head pose estimation. Internally it uses   dlib (so, the ERT method) for landmark localisation. However, it performs tracking instead of per-frame face-detection + localisation thus it should run much faster than the vanilla dlib implementation. Moreover, we trained some additional 49 and 68-landmark models on a larger dataset. Specifically, the following tools are provided:
