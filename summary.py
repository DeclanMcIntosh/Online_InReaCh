from pathlib import Path
import numpy as np

results = {}
for path in Path('Experiments/1c8bb67135c85f58b6f821336810030c8446ea11/').rglob('*.csv'):
    print(path)
    experiment = str(path).split('/')[-3]
    if not experiment in results.keys(): results[experiment] = {}
    file1 = open(path, 'r')
    Lines = file1.readlines()
    for line in Lines:
        try:
            if not line.split(',')[1] in results[experiment].keys():
                results[experiment][line.split(',')[1]] = [float(line.split(',')[0])] 
            else: results[experiment][line.split(',')[1]].append(float(line.split(',')[0]))
        except Exception:
            pass



# inference speed, update speed, image_wise_predictions, image_wise_actual, pixel_wise_AUROC, channel_percision, num_channels, feature_extraction_time, image_wise_AUROC 
keys = list(results.keys()) 
keys.sort()
for key in keys:
    outstr = key+','
    for key_sub in results[key].keys():
        outstr +=  str(np.mean(results[key][key_sub][:50])) + ','
    print(outstr)


# pixel, image 
# mvtec 0.953, 0.940
# bean 0.984, 0.945
# 