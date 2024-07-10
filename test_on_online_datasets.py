import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"



os.system("python OnlineInReaCh.py -pos_w 0.0 -ttl 10 -minl 2 -seed 0 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/mvtec_anomaly_detection/ -n ___ABLATION_NO_POS")


#os.system("python OnlineInReaCh.py -pos_w 0.0 -ttl 40 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa/            -n 40_online_test_visa_0_all_anomalies_first")
#os.system("python OnlineInReaCh.py -pos_w 0.0 -ttl 40 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/BTech_Dataset_transformed/       -n 40_online_test_beantech_0_all_anomalies_first")
#os.system("python OnlineInReaCh.py -pos_w 0.0 -ttl 80 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa/    -n 80_online_visa_0")

#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 80 -minl 2 -seed 11235813 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa_40/ -n 80__n_seed_pre_trained_visa_40")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 40 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa_40/ -n 40_pre_trained_visa_40")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 150 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa_40/ -n 40_pre_trained_visa_40")

##os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 40 -minl 2 -seed 1 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa_40/ -n _seed_1_40_pre_trained_visa_40")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 40 -minl 2 -seed 2 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa_40/ -n _seed_2_40_pre_trained_visa_40")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 40 -minl 2 -seed 3 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa_40/ -n _seed_3_40_pre_trained_visa_40")

#os.system("python OnlineInReaCh.py --pretrain -tur 1 -pos_w 0.0 -ttl 80 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa/    -n dur_test_updates_pre_trained_visa_0")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 40 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa_10/ -n pre_trained_visa_10")
#os.system("python OnlineInReaCh.py --pretrain -tur 1 -pos_w 0.0 -ttl 80 -minl 2 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa_40/ -n dur_test_updates_pre_trained_visa_40")

#os.system("python OnlineInReaCh.py -pos_w 0.0 -ttl 80 -minl 2 -seed 0 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa/    -n seed_0_80_online_visa_0")
#os.system("python OnlineInReaCh.py -pos_w 0.0 -ttl 80 -minl 2 -seed 1 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/visa/    -n seed_1_80_online_visa_0")



#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 10 -minl 2 -tur 1 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/MVTEC_AUGMENTED_TEST_DATA_NON_STATIONARY/piecewise/       -n qual_gen_piecewise")
#os.system("python OnlineInReaCh.py --pretrain -pos_w 0.0 -ttl 10 -minl 2 -tur 1 -seed 112358 -data_dir /home/paul/Desktop/Declan/Testing_Unsupervised_InReaCh_Competators/MVTEC_AUGMENTED_TEST_DATA_NON_STATIONARY/transpose/       -n qual_gen_transpose")
