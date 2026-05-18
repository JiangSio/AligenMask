import os

root_dir = "/data1/gpt/jtj/supp_exp/mvtec+visa"
save_file = "name-anomaly.txt"
if os.path.exists(save_file):
    os.remove(save_file)

clsses = [x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
clsses.sort()
for clss_name in clsses:
    clss_dir = os.path.join(root_dir, clss_name)
    test_path = os.path.join(clss_dir, "test")
    anomalies = [x for x in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, x))]
    anomalies.remove("good")
    anomalies.sort()
    for anomaly in anomalies:
        with open(save_file, "a") as f:
            if f.seek(0, os.SEEK_END) > 0:
                f.write("\n")
            f.write(clss_name + "+" + anomaly)
        