class Config:
    ####################Please modify all below path to get the model working######################
    repository_path ='/home/geppetto/jie/haobug/dlib_and_chehra_stuff'
    ert_model_path ='/home/geppetto/jie/haobug/dlib_and_chehra_stuff/models/new3_68_pts_UAD_1_tr_6_cas_15.dat'
    auxiliary_model_path ='/home/geppetto/jie/haobug/dlib_and_chehra_stuff/models/additional_svrs.model'
    ibug_face_tracker_path = '/home/geppetto/jie/haobug/dlib_and_chehra_stuff/ibug_face_tracker/'
    hao_emotion_recognition_model_path='/home/geppetto/jie/haobug/dupe_net_8.th'
    overlap_threshold = 0.64
    hard_failure_threshold = -0.5
    soft_failure_threshold = 0.0
    maximum_number_of_soft_failures = 2
    
    use_gamma_correction = 1
