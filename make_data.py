import os

trainval_txt_path = "/Users/xuan/Downloads/annotations/trainval.txt"
test_txt_path = "/Users/xuan/Downloads/annotations/test.txt"
image_folder = "/Users/xuan/Downloads/images/"
target_folder_train = "/Users/xuan/Downloads/images/train/"
target_folder_test = "/Users/xuan/Downloads/images/test/"

trainval_txt = open(trainval_txt_path)
lines = trainval_txt.readlines()
trainval_txt.close()
for line in lines:
    l = line.split()
    curr_filename = l[0] + ".jpg"
    curr_class = l[1]
    if not os.path.isdir(target_folder_train + curr_class):
        os.mkdir(target_folder_train + curr_class)
    os.rename(
        image_folder + curr_filename, 
        target_folder_train + curr_class + "/" + curr_filename
    )

test_txt = open(test_txt_path)
lines = test_txt.readlines()
test_txt.close()
for line in lines:
    l = line.split()
    curr_filename = l[0] + ".jpg"
    curr_class = l[1]
    if not os.path.isdir(target_folder_test + curr_class):
        os.mkdir(target_folder_test + curr_class)
    os.rename(
        image_folder + curr_filename, 
        target_folder_test + curr_class + "/" + curr_filename
    )
