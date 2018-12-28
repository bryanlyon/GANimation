import os

csv_train = open('train_ids.csv', 'w')
csv_test = open('test_ids.csv', 'w')

file_list = os.listdir('./imgs_128')

trian_file_list = file_list[0:int(len(file_list)/10*9)]
test_file_list = file_list[int(len(file_list)/10*9):]

trian_file_list = "\n".join(trian_file_list)
test_file_list = "\n".join(test_file_list)

csv_test.write(test_file_list)
csv_train.write(trian_file_list)


csv_test.close()
csv_train.close()
