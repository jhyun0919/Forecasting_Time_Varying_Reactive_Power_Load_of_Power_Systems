function [RMSE, MAPE, MAE] = accuracy(testY, pred)

RMSE = sqrt(mse(testY - pred));

MAPE_total = mape(testY, pred);
MAPE = mean(MAPE_total);

MAE_total = mae(testY, pred);
MAE = mean(MAE_total);
