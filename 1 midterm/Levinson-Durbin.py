import time
from datetime import datetime
import numpy as np
from statsmodels.tsa.stattools import levinson_durbin


def parser(n=None):
    '''
    This function parse input file
    :param n: number of lines (if None -> all)
    :return: dates (array of dates), data_list (array of the Appliances column)
    '''

    path = r"C:\Users\giaco\Documents\UNIPI\I ANNO\ISPR\1 midterm\energydata_complete.csv"

    with open(path, "r") as file:
        file.readline()
        line = file.readline()
        matrix=[]
        dates=[]
        count = 0
        while(line!=""):
            count += 1
            line = line.replace('"', '')
            raw_elements = line.strip().split(",")
            dates.append(datetime.strptime(raw_elements[0], '%Y-%m-%d %H:%M:%S'))
            matrix.append([float(x) for x in raw_elements[1:]])
            if n is not None and count == n:
                break
            line = file.readline()
    data = np.array(matrix)[:, 0]
    data_list = data.tolist()
    return dates, data_list


def train(train_data, k):
    '''
    Find the coefficients (len(coefficients) = k) for the time serie train_data
    :param train_data: the time series
    :param k: order of the AR model (number of coefficients)
    :return: array of coefficients
    '''
    # isacov need to be set to true if the first argument contains the autocovariance
    sigmav, arcoefs, pacf, sigma, phi = levinson_durbin(train_data, nlags=k, isacov=False)
    return arcoefs

if __name__ == '__main__':
    dates, data = parser()

    # (91 days) * 24 * (6 data per day)
    train_limit = (91 * 24 * 6) + 1

    print("Training set {} -> {}".format(dates[0], dates[train_limit - 1]))
    print("Test set {} -> {}".format(dates[train_limit], dates[-1]))

    train_dates = dates[:train_limit]
    train_data = data[:train_limit]

    test_dates = dates[train_limit:]
    test_data = data[train_limit:]

    k = ar_order = 4

    coefficients = train(train_data, k)

    estimations=[]
    errors=[]

    #write each prediction on file
    filename = "ARMA({},{})_levinson".format(ar_order, 0)

    with open("output/{}".format(filename), "w") as file:
        start_time = time.time()
        n_retrains = 0
        for i in range(train_limit, len(data)):
            estimation = 0
            #computing the predection
            for idx, coeff in enumerate (coefficients):
                estimation += data[i-(idx+1)] * coeff
            estimations.append(estimation)
            #calc the error
            error = abs(data[i] - estimation)
            errors.append(error)
            #update train_data
            train_data.append(data[i])
            #and re train
            coefficients = train(train_data, k)

        end_time = time.time()
        # last thre lines of the file will contain:
        # time elpased from the beginning
        # MAE
        # number of retrains
        file.write("{}\n".format(end_time - start_time))
        file.write("{}\n".format(np.mean(np.array(errors))))
        file.write("{}\n".format(len(data)- train_limit))






