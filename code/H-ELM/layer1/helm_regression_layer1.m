function [Training_time,Testing_time, TrainingAccuracy_RMSE, TestingAccuracy_RMSE, TrainingAccuracy_MAPE, TestingAccuracy_MAPE, TrainingAccuracy_MAE, TestingAccuracy_MAE, predict_yy] = helm_regression_layer2(TrainingData, TestingData, No_of_Output, N1)




%%%%%%%%%%% Load training dataset
train_y=TrainingData(:,1:No_of_Output)';                 %   training label data
train_x=TrainingData(:,No_of_Output+1:size(TrainingData,2))'; %   training feature data
clear TrainingData;                                %   Release raw training data array

%%%%%%%%%%% Load testing dataset

test_y=TestingData(:,1:No_of_Output)';               %   testing label data
test_x=TestingData(:,No_of_Output+1:size(TestingData,2))'; %   testing feautre data
clear TestingData;                                 %   Release raw testing data array

%%%%%%%%%%% Set hyper-parameters

N=N1+1;
b1=2*rand(size(train_x',2)+1,N1)-1;
b=orth(2*rand(N1+1,N)'-1)';
C = 2^-30; s = .8;


%%%%%%%%%%%%%%%%%%%%%%%
% training part
%%%%%%%%%%%%%%%%%%%%%%%

tic
train_x = zscore(train_x)';

%% 1st layer RELM
[T1, beta1, ps1] = relm_train(train_x, b1);
clear train_x; clear b1;
fprintf(1,'Layer 1: Max Val of Output %f Min Val %f\n',max(T1(:)),min(T1(:)));

%% 2nd layer RELM
%[T2, beta2, ps2] = relm_train(T1, b2);
%clear T1; clear b2;
%fprintf(1,'Layer 2: Max Val of Output %f Min Val %f\n',max(T2(:)),min(T2(:)));

%% 3rd layer RELM
%[T3, beta3, ps3] = relm_train(T2, b3);
%clear T2; clear b3;
%fprintf(1,'Layer 3: Max Val of Output %f Min Val %f\n',max(T3(:)),min(T3(:)));

%% 4rd layer RELM
%[T4, beta4, ps4] = relm_train(T3, b4);
%clear T3; clear b4;
%fprintf(1,'Layer 4: Max Val of Output %f Min Val %f\n',max(T4(:)),min(T4(:)));

%% Original ELM regressor
[T, l, beta] = elm_train(train_y, T1, b, s, C);
clear T1;
fprintf(1,'ELM Layer: Max Val of Output %f Min Val %f\n',l,min(T(:)));

%% Finsh Training
Training_time = toc;
%% Calculate the training accuracy
predict_y = (T * beta)';
clear T;

disp('Training has been finished!');
disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);

[TrainingAccuracy_RMSE, TrainingAccuracy_MAPE, TrainingAccuracy_MAE] = accuracy(train_y, predict_y);
clear H;

disp(['Training Accuracy RMSE= ', num2str(TrainingAccuracy_RMSE)]);
disp(['Training Accuracy MAPE= ', num2str(TrainingAccuracy_MAPE)]);
disp(['Training Accuracy MAE= ', num2str(TrainingAccuracy_MAE)]);



%%%%%%%%%%%%%%%%%%%%%%%
% testing part
%%%%%%%%%%%%%%%%%%%%%%%

tic;
test_x = zscore(test_x)';

%% 1st layer feedforward
TT1 = relm_test(test_x, beta1, ps1);
clear test_x;

%% 2nd layer feedforward
%TT2 = relm_test(TT1, beta2, ps2);
%clear TT1;

%% 3rd layer feedforward
%TT3 = relm_test(TT2, beta3, ps3);
%clear TT2;

%% 4th layer feedforward
%TT4 = relm_test(TT3, beta4, ps4);
%clear TT3;

%% Last layer feedforward
predict_yy = (elm_test(TT1, beta, l, b))';
clear TT1; clear b;

%% Calculate the testing accuracy
Testing_time = toc;
[TestingAccuracy_RMSE, TestingAccuracy_MAPE, TestingAccuracy_MAE] = accuracy(test_y, predict_yy);

disp('Testing has been finished!');
disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);

disp(['Testing Accuracy RMSE= ', num2str(TestingAccuracy_RMSE)]);
disp(['Testing Accuracy MAPE= ', num2str(TestingAccuracy_MAPE)]);
disp(['Testing Accuracy MAE= ', num2str(TestingAccuracy_MAE)]);

fprintf('\n');
