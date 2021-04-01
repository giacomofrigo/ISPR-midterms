from datetime import datetime
import time

import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima.model import ARIMA



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

    data = np.array(matrix)[:,0]
    data_list = data.tolist()
    return dates,data_list

if __name__ == '__main__':
    dates, data = parser()

    # (91 days) * 24 * (6 data per day)
    train_limit = len(data) - (24*6)

    print("Training set {} -> {}".format(dates[0], dates[train_limit-1]))
    print("Test set {} -> {}".format(dates[train_limit], dates[-1]))

    train_dates = dates[:train_limit]
    train_data = data[:train_limit]

    test_dates = dates[train_limit:]
    test_data = data[train_limit:]

    #plotting acf and pacf
    plot_pacf(train_data, lags=50)
    plt.show()
    plot_acf(train_data, lags=50)
    plt.show()

    #ar model order
    ar_order = 2
    ma_order = 0

    #create the ARIMA model
    ar = ARIMA(train_data, order=(ar_order, 0, ma_order))

    #fitting the model
    model = ar.fit()
    #print fitting results
    #print(model.summary())

    steps=1

    errors = []
    error_threshold = 0

    #this flag will enable the retraining of the model once error_threshold is exceeded
    retrain = True

    #write on file each prediction
    filename = "ARMA({},{})_{}".format(ar_order, ma_order, error_threshold)

    with open("output_last_day/{}".format(filename), "w") as file:

        predictions=[]
        n_retrains = 0
        start_time = time.time()

        for idx in range(train_limit, len(data)):
            #make steps predictions
            prediction = model.forecast(steps=steps)[-1]
            predictions.append(prediction)
            #calc the error
            error = abs(data[idx]-prediction)
            errors.append(error)
            #update train data
            train_data.append(data[idx])
            file.write("{},{},{},{}\n".format(prediction, data[idx],error, model.arparams))
            #if retrain is true and error large -> retrain the system
            if retrain and error > error_threshold:
                print ("Retraining at idx {} Prediction: {}, Real: {}".format(idx, prediction, data[idx]))
                ar = ARIMA(train_data, order=(ar_order, 0, ma_order))
                model = ar.fit()
                coeff = model.arparams
                steps = 1
                n_retrains += 1
            else:
                if retrain:
                    steps += 1


        end_time=time.time()
        #last thre lines of the file will contain:
        #time elpased from the beginning
        #MAE
        #number of retrains
        file.write("{}\n".format(end_time-start_time))
        file.write("{}\n".format(np.mean(np.array(errors))))
        file.write("{}\n".format(n_retrains))

    #creating an array with only the error mean (errors time) for graph purposes
    error_mean=[]
    for error in errors:
        error_mean.append(np.mean(np.array(errors)))

    #saving the result through some graphs
    plt.plot(test_dates, errors, label="Absolute Error")
    plt.plot(test_dates, error_mean, label="Absolute Error Mean")
    plt.title('ARMA({},{}), Error {} - Absolute Error'.format(ar_order, ma_order, error_threshold))
    plt.xlabel('Time')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.savefig("output/plot_{}_errors".format(filename))
    plt.close()

    plt.plot(test_dates[-24 * 6:], test_data[-24 * 6:], label="Real Values")
    plt.plot(test_dates[-24 * 6:], predictions[-24 * 6:], label="Predictions")
    plt.title('ARMA({},{}), Error {} - Last day'.format(ar_order, ma_order, error_threshold))
    plt.xlabel('Time')
    plt.ylabel('Appliances')
    plt.legend()
    plt.savefig("output/plot_{}_lastday".format(filename))
    plt.close()

    plt.plot(test_dates, test_data,label="Real Values")
    plt.plot(test_dates, predictions,label="Predictions")
    plt.title('ARMA({},{}), Error {} - Test Set'.format(ar_order, ma_order, error_threshold))
    plt.xlabel('Time')
    plt.ylabel('Appliances')
    plt.legend()
    plt.savefig("output/plot_{}_allpredictions".format(filename))
    plt.close()