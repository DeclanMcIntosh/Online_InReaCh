import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"


os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 10 -minl 2 -tur 1 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/MVTEC_AUGMENTED_TEST_DATA_NON_STATIONARY/simulate_spot_light/  -n __no_spatial_loss")
##

os.system("python OnlineInReaCh.py -pos_w 0.0 -ttl 40 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators//mvtec_anomaly_detection/        -n 40_online_test_mvtec_0_all_anomalies_first")

#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.025 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.025_ttl_15_mil_2")
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.05 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.05_ttl_15_mil_2")
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.075 -seed 112,0.7735533475875854358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.075_ttl_15_mil_2")
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.125 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.125_ttl_15_mil_2")
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.15 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.15_ttl_15_mil_2")


#os.system("python OnlineInReaCh.py --pretrain -tur 1 -pos_w 0.0 -ttl 40 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/mvtec_anomaly_detection/    -n dur_test_updates_pre_trained_mvtec_0")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 40 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/mvtec_anomaly_detection_10/ -n pre_trained_mvtec_10")
#os.system("python OnlineInReaCh.py --pretrain -tur 1 -pos_w 0.0 -ttl 40 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/mvtec_anomaly_detection_40/ -n dur_test_updates_pre_trained_mvtec_40")

#os.system("python OnlineInReaCh.py --pretrain -tur 1 -pos_w 0.0 -ttl 40 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/BTech_Dataset_transformed/    -n dur_test_updates__pre_trained_beantech_0")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 40 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/BTech_Dataset_transformed_10/ -n pre_trained_beantech_10")
#os.system("python OnlineInReaCh.py --pretrain -tur 1 -pos_w 0.0 -ttl 40 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/BTech_Dataset_transformed_40/ -n dur_test_updates__pre_trained_beantech_40")

#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 80 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa/    -n 80_pre_trained_visa_0")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 40 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa_10/ -n pre_trained_visa_10")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 80 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa_40/ -n 80_pre_trained_visa_40")


#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 10 -minl 2 -tur 1 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/MVTEC_AUGMENTED_TEST_DATA_NON_STATIONARY/gamma_0_25/    -n non_stationary_gamma_0_25")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 10 -minl 2 -tur 1 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/MVTEC_AUGMENTED_TEST_DATA_NON_STATIONARY/hue_90/        -n non_stationary_hue_90")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 10 -minl 2 -tur 1 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/MVTEC_AUGMENTED_TEST_DATA_NON_STATIONARY/brightness_64/ -n non_stationary_brightness_64")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 10 -minl 2 -tur 1 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/MVTEC_AUGMENTED_TEST_DATA_NON_STATIONARY/hue_175/       -n non_stationary_hue_175")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 10 -minl 2 -tur 1 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/MVTEC_AUGMENTED_TEST_DATA_NON_STATIONARY/90_degree_rotate/       -n 90_degree_rotate")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 10 -minl 2 -tur 1 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/MVTEC_AUGMENTED_TEST_DATA_NON_STATIONARY/red_box/       -n red_box")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 10 -minl 2 -tur 1 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/MVTEC_AUGMENTED_TEST_DATA_NON_STATIONARY/transpose/       -n transpose")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 10 -minl 2 -tur 1 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/MVTEC_AUGMENTED_TEST_DATA_NON_STATIONARY/piecewise/       -n piecewise")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 10 -minl 2 -tur 99999 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/MVTEC_AUGMENTED_TEST_DATA_NON_STATIONARY/hard_test_test_split/       -n hard_test_test_split")


#os.system("python OnlineInReaCh.py -pos_w 0.0 -ttl 40 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/mvtec_anomaly_detection_40/    -n qual_gen_mvtec_40_online")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 10 -minl 2 -tur 1 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/MVTEC_AUGMENTED_TEST_DATA_NON_STATIONARY/red_box/       -n qual_gen_red_box")