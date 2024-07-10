import os



# Optimal Test
#os.system("python OnlineInReaCh.py -ttl 40 -minl 2 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_proposed_optimal_ttl_40_mil_2_no_positional")

# Positional embedding hyperparameter search
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.001 -seed 112358 -data_dir data/mvtec_anomaly_detection/    -n hyperparameter_serach_positional_score_0.001_ttl_15_mil_2")
os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.025 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.025_ttl_15_mil_2")
os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.05 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.05_ttl_15_mil_2")
os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.075 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.075_ttl_15_mil_2")
os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.125 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.125_ttl_15_mil_2")
os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.15 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.15_ttl_15_mil_2")
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.1 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.1_ttl_15_mil_2")
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.2 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.2_ttl_15_mil_2")
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.3 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.3_ttl_15_mil_2")
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.4 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.4_ttl_15_mil_2")
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.5 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.5_ttl_15_mil_2")
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.6 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.6_ttl_15_mil_2")
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.7 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.7_ttl_15_mil_2")
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.8 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.8_ttl_15_mil_2")
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.9 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.9_ttl_15_mil_2")
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 1.0 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_1.0_ttl_15_mil_2")
#os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -pos_w 0.0 -seed 112358 -data_dir data/mvtec_anomaly_detection/      -n hyperparameter_serach_positional_score_0.0_ttl_15_mil_2")


if False:  # previous test no 
    os.system("python OnlineInReaCh.py -ttl 15 -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_baseline")

    os.system("python OnlineInReaCh.py -ttl 2  -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl__2 ")
    os.system("python OnlineInReaCh.py -ttl 3  -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl__3 ")
    os.system("python OnlineInReaCh.py -ttl 4  -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl__4 ")
    os.system("python OnlineInReaCh.py -ttl 5  -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl__5 ")
    os.system("python OnlineInReaCh.py -ttl 6  -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl__6 ")
    os.system("python OnlineInReaCh.py -ttl 7  -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl__7 ")
    os.system("python OnlineInReaCh.py -ttl 10 -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl_10")
    os.system("python OnlineInReaCh.py -ttl 12 -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl_12")
    os.system("python OnlineInReaCh.py -ttl 15 -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl_15")
    os.system("python OnlineInReaCh.py -ttl 20 -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl_20")
    os.system("python OnlineInReaCh.py -ttl 25 -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl_25")
    os.system("python OnlineInReaCh.py -ttl 30 -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl_30")
    os.system("python OnlineInReaCh.py -ttl 35 -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl_35")
    os.system("python OnlineInReaCh.py -ttl 40 -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl_40")
    os.system("python OnlineInReaCh.py -ttl 50 -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl_50")
    os.system("python OnlineInReaCh.py -ttl 75 -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_ttl_75")


    os.system("python OnlineInReaCh.py -ttl 15 -minl 2 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_minl_2")
    os.system("python OnlineInReaCh.py -ttl 15 -minl 3 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_minl_3")
    os.system("python OnlineInReaCh.py -ttl 15 -minl 4 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_minl_4")
    os.system("python OnlineInReaCh.py -ttl 15 -minl 5 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_minl_5")
    os.system("python OnlineInReaCh.py -ttl 15 -minl 6 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_minl_6")
    os.system("python OnlineInReaCh.py -ttl 15 -minl 7 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_minl_7")
    os.system("python OnlineInReaCh.py -ttl 15 -minl 8 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_minl_8")
    os.system("python OnlineInReaCh.py -ttl 15 -minl 9 -seed 112358 -data_dir data/mvtec_anomaly_detection/ -n hyperparameter_serach_final_minl_9")
