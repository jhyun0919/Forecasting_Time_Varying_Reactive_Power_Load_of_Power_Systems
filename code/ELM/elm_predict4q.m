load '/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/train4q.mat';
load '/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/test4q.mat';
write_file='/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/elm_iteration_result4q.csv';

NumberofHiddenNeurons=25;
No_of_Output=24;
ActivationFunction='sig';

RMSE_Training = [];
RMSE_Testing = [];
MAPE_Training = [];
MAPE_Testing = [];
MAE_Training = [];
MAE_Testing = [];

for x = 1:1
x
[TrainingTime, TestingTime, TrainingAccuracy_RMSE, TestingAccuracy_RMSE, TrainingAccuracy_MAPE, TestingAccuracy_MAPE, TrainingAccuracy_MAE, TestingAccuracy_MAE, predict_q] = ELM_MultiOutputRegression(train_q, test_q, No_of_Output, NumberofHiddenNeurons, ActivationFunction);
TrainingTimeList(x,:) = TrainingTime;

RMSE_Training(x,:) = TrainingAccuracy_RMSE;
RMSE_Testing(x,:) = TestingAccuracy_RMSE;
MAPE_Training(x,:) = TrainingAccuracy_MAPE;
MAPE_Testing(x,:) = TestingAccuracy_MAPE;
MAE_Training(x,:) = TrainingAccuracy_MAE;
MAE_Testing(x,:) = TestingAccuracy_MAE;
end

AccList = horzcat(RMSE_Training, RMSE_Testing, MAPE_Training, MAPE_Testing, MAE_Training, MAE_Testing);
headers = {'RMSE_Train', 'RMSE_Test', 'MAPE_Train', 'MAPE_Test', 'MAE_Train', 'MAE_Test'};
csvwrite_with_headers(write_file,AccList,headers);
TrainingtTime = mean(TrainingTimeList)

predict_q = predict_q';
pathname = fileparts('/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/');
matfile = fullfile(pathname, 'elm_predict4q.mat');
save(matfile, 'predict_q');
clear;