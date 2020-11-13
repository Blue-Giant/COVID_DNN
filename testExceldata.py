import csv
import xlrd, xlwt   #xlwt只能写入xls文件
import numpy as np


def load_csvData(datafile=None):
    csvdata_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for csvdata in csvreader:
        if str.isnumeric(csvdata[1]):
            csvdata_list.append(int(csvdata[1]))
            csvdate_list.append(icount)
            icount = icount + 1
    return csvdate_list, csvdata_list


def randSample_existData(data1, data2, batchsize=1):
    data1_samples = []
    data2_samples = []
    data_length = len(data1)
    indexes = np.random.randint(data_length+1, size=batchsize)
    for i_index in indexes:
        data1_samples.append(data1[i_index])
        data2_samples.append(data2[i_index])
    return data1_samples, data2_samples


if __name__ == "__main__":
    filename = 'data2csv/Italia_data.csv'
    date, data = load_csvData(filename)
    print(date)
    print('\n')
    print(data)

    index = np.random.randint(5, size=2)
    print(index)

    data1samples, data2samples = randSample_existData(date, data, batchsize=2)
    print(data1samples)
    print(data2samples)