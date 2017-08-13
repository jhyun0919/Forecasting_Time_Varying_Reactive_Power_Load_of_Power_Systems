load '/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/train_p.mat';
load '/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/val_p.mat';;
write_file = '/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/elm_optimal.csv';

%%%%%

No_of_Output=24;
ActivationFunction='sig';
NumberofHiddenNeurons=1;

RMSE_List = [];
MAPE_List = [];
%MAE_List = [];

for x = 1:150
    RMSE_temp = [];
    MAPE_temp = [];
    MAE_temp = [];
    for xx = 1:30
        [TrainingTime, TestingTime, TrainingAccuracy_RMSE, TestingAccuracy_RMSE, TrainingAccuracy_MAPE, TestingAccuracy_MAPE, TrainingAccuracy_MAE, TestingAccuracy_MAE, predict_p] = ELM_MultiOutputRegression(train_p, val_p, No_of_Output, NumberofHiddenNeurons, ActivationFunction);
        RMSE_temp(xx,:) = TestingAccuracy_RMSE;
        MAPE_temp(xx,:) = TestingAccuracy_MAPE;
        MAE_temp(xx,:) = TestingAccuracy_MAE;
    end

    RMSE_List(x,:) = mean(RMSE_temp);
    MAPE_List(x,:) = mean(MAPE_temp);
    MAE_List(x,:) = mean(MAE_temp);
    NumberofHiddenNeurons = NumberofHiddenNeurons + 1;

end

AccList = horzcat(RMSE_List, MAPE_List, MAE_List);
headers = {'RMSE', 'MAPE', 'MAE'};
csvwrite_with_headers(write_file,AccList,headers)
