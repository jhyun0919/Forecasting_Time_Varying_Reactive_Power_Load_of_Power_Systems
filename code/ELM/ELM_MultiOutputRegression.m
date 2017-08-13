function [TrainingTime, TestingTime, TrainingAccuracy_RMSE, TestingAccuracy_RMSE, TrainingAccuracy_MAPE, TestingAccuracy_MAPE, TrainingAccuracy_MAE, TestingAccuracy_MAE] = elm_MultiOutputRegression(TrainingData, TestingData, No_of_Output, NumberofHiddenNeurons, ActivationFunction)

%
% Reference : http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
%
    %%%%    Authors:    Park Jee Hyun
    %%%%    SUNGKYUNKWAN UNIVERSITY, KOREA [UNDERGRADUATE STUDENT]
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE [EXCHANGE STUDENT]
    %%%%    EMAIL:      jhyun19@gmail.com
    %%%%    WEBSITE:    https://jhyun0919.github.io
    %%%%    DATE:       AUG 2017


disp(['Number of HiddenNeurons = ', num2str(NumberofHiddenNeurons)]);

%%%%%%%%%%% Load training dataset
T=TrainingData(:,1:No_of_Output)';                 %   training label data
P=TrainingData(:,No_of_Output+1:size(TrainingData,2))'; %   training feature data
clear TrainingData;                                %   Release raw training data array

%%%%%%%%%%% Load testing dataset

TV.T=TestingData(:,1:No_of_Output)';               %   testing label data
TV.P=TestingData(:,No_of_Output+1:size(TestingData,2))'; %   testing feautre data
clear TestingData;                                 %   Release raw testing data array

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                TRAINING                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
clear P;                                            %   Release input of training data
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = hardlim(tempH);
        %%%%%%%% More activation functions can be added here
    case {'relu'}
        %%%%%%%% RelU (still need to be checked... not working currently...)
        H = relu(tempH);
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight=pinv(H') * T';
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;        %   Calculate CPU time (seconds) spent for training ELM

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data

TrainingAccuracy_RMSE=sqrt(mse(T - Y));              %   Calculate training accuracy (RMSE) for regression case
[TrainingAccuracy_MAPE, TrainingAccuracy_MAE] = mape_mae(T, Y);
clear H;

disp(['    Training Time = ', num2str(TrainingTime), ' seconds' ]);
disp(['    Training Accuracy RMSE= ', num2str(TrainingAccuracy_RMSE)]);
disp(['    Training Accuracy MAPE= ', num2str(TrainingAccuracy_MAPE)]);
disp(['    Training Accuracy MAE= ', num2str(TrainingAccuracy_MAE)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  TESTING                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);
        %%%%%%%% More activation functions can be added here
end
TY=(H_test' * OutputWeight)';                         %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test;            %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
TestingAccuracy_RMSE=sqrt(mse(TV.T - TY));            %   Calculate testing accuracy (RMSE) for regression case
[TestingAccuracy_MAPE, TestingAccuracy_MAE] = mape_mae(TV.T, TY);

clear TV.T;


disp(['    Testing Time = : ', num2str(TestingTime), ' seconds' ]);
disp(['    Testing Accuracy RMSE= ', num2str(TestingAccuracy_RMSE)]);
disp(['    Testing Accuracy MAPE= ', num2str(TestingAccuracy_MAPE)]);
disp(['    Testing Accuracy MAE= ', num2str(TestingAccuracy_MAE)]);
