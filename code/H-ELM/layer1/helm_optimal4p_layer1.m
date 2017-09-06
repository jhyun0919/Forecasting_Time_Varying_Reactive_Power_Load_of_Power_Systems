load '/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/train_p.mat';
load '/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/val_p.mat';;
write_file = '/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/helm_optimal_layer1.csv';

%%%%%

No_of_Output=24;
ActivationFunction='sig';
NumberofHiddenNeurons = 1;



RMSE_List = [];
MAPE_List = [];
MAE_List = [];

for x = 1:150
    NumberofHiddenNeurons_layer_1=NumberofHiddenNeurons;
    RMSE_temp = [];
    MAPE_temp = [];
    MAE_temp = [];

    for xx = 1:30
        x
        [Training_time,Testing_time, TrainingAccuracy_RMSE, TestingAccuracy_RMSE, TrainingAccuracy_MAPE, TestingAccuracy_MAPE, TrainingAccuracy_MAE, TestingAccuracy_MAE, predict_p] = helm_regression_layer1(train_p, val_p, No_of_Output, NumberofHiddenNeurons_layer_1);
        RMSE_temp(xx,:) = TestingAccuracy_RMSE;
        MAPE_temp(xx,:) = TestingAccuracy_MAPE;
        MAE_temp(xx,:) = TestingAccuracy_MAE;
    end

    RMSE_List(x,:) = mean(RMSE_temp);
    MAPE_List(x,:) = mean(MAPE_temp);
    MAE_List(x,:) = mean(MAE_temp);
    NumberofHiddenNeurons = NumberofHiddenNeurons + 1;

end

AccList = horzcat(RMSE_List, MAPE_List,MAE_List);
headers = {'RMSE', 'MAPE', 'MAE'};
csvwrite_with_headers(write_file,AccList,headers);
clear;
