load '/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/train_q.mat';
load '/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/val_q.mat';;
write_file = '/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/helm_optimal.csv';

%%%%%

No_of_Output=24;
ActivationFunction='sig';
NumberofHiddenNeurons_layer_1=1;
NumberofHiddenNeurons_layer_2=1;

RMSE_List = [];
MAPE_List = [];
MAE_List = [];

for x = 1:300
    RMSE_temp = [];
    MAPE_temp = [];
    MAE_temp = [];
    for xx = 1:20
    	 x
       N1=NumberofHiddenNeurons;
		   N=N1+1;
    	 b1=2*rand(size(train_x',2)+1,N1)-1;
		   b=orth(2*rand(N1+1,N)'-1)';
		   C = 2^-30; s = .8;

       [Training_time,Testing_time, TrainingAccuracy_RMSE, TestingAccuracy_RMSE, TrainingAccuracy_MAPE, TestingAccuracy_MAPE, TrainingAccuracy_MAE, TestingAccuracy_MAE] = helm_regression_01(train_x, train_y, test_x, test_y, b1, b, s, C);
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
csvwrite_with_headers(Result_File,AccList,headers)
