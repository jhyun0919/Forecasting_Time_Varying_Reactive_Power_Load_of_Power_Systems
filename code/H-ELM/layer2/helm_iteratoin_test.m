load '/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/train_p.mat';
load '/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/test_p.mat';
write_file='/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/helm_iteration_result_p.csv';

No_of_Output=24;
ActivationFunction='sig';
NumberofHiddenNeurons = 1;
NumberofHiddenNeurons_layer_1=25;
NumberofHiddenNeurons_layer_2=25;



RMSE_Training = [];
RMSE_Testing = [];
MAPE_Training = [];
MAPE_Testing = [];
MAE_Training = [];
MAE_Testing = [];

for x = 1:1
x
[Training_time,Testing_time, TrainingAccuracy_RMSE, TestingAccuracy_RMSE, TrainingAccuracy_MAPE, TestingAccuracy_MAPE, TrainingAccuracy_MAE, TestingAccuracy_MAE, predict_p] = helm_regression_layer2(train_p, test_p, No_of_Output, NumberofHiddenNeurons_layer_1, NumberofHiddenNeurons_layer_2);
TrainingTimeList(x,:) = Training_time;
RMSE_Training(x,:) = TrainingAccuracy_RMSE;
RMSE_Testing(x,:) = TestingAccuracy_RMSE;
end

AccList = horzcat(RMSE_Training, RMSE_Testing);
headers = {'Training', 'Testing'};
csvwrite_with_headers(write_file,AccList,headers);
TrainintTime = mean(TrainingTimeList);

predict_p = predict_p';
pathname = fileparts('/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/');
matfile = fullfile(pathname, 'helm_2_predict_p.mat');
save(matfile, 'predict_p');
