import os

root_dir = "/data1/gpt/jtj/asynthesis_data"
save_file = "name-list.txt"
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

    with open(save_file, "a") as f:
        f.write(f"\t\"{clss_name}\", \n")

    for anomaly in anomalies:
        pass
        