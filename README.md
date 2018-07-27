# haobug

This is project create to recognize facial emotion through camera.

The author of this project is Chunliang Hao and Jie Shen

# Develop Environment
Following environments(or compatible ones) are required to run this project:
1. pytorch 0.40
2. cuda 9.0
3. cudnn 7.1.2
4. opencv-python 3.4.1.16
5. numpy 1.14.5
6. dlib 19.4

# Dependency

You will need the following base projects and models to run this project (Contact Jie Shen or Chunliang Hao for the access of both models):

1. dlib_and_chehra_stuff     

    A repository of code that performs facial landmark tracking, eye point localisation, and head pose estimation. Internally it uses   dlib (so, the ERT method) for landmark localisation. However, it performs tracking instead of per-frame face-detection + localisation thus it should run much faster than the vanilla dlib implementation. Moreover, we trained some additional 49 and 68-landmark models on a larger dataset. Specifically, the following tools are provided:

2. dupe_net_8
    
    A model trained  to perform facial emotion recognition.
    
# Configuring Before Run
    
Please make sure all directories are changed to your local path in AllConfigs.py before run the project.

Please change the id of GPU in your computer in haobug_v1.py by modifing the integer in this sentence: os.environ['CUDA_VISIBLE_DEVICES'] = '1'. 
Change the number to 0 if you only have one GPU. 

Just run the haobug_v1.py file, no args needed.