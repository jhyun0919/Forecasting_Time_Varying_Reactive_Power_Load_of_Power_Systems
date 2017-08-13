function [Fittness,OutputWeight] = ELM_X (Elm_Type, weight_bias, P, T, TV, NumberHiddenNeurons)
NumberInputNeurons=size(P, 1);
NumberofTrainingData=size(P, 2);
NumberofTestingData=size(TV.P, 2);
Gain=1;
temp_weight_bias=reshape(weight_bias, NumberHiddenNeurons, NumberInputNeurons+1);
InputWeight=temp_weight_bias(:, 1:NumberInputNeurons);
BiasofHiddenNeurons=temp_weight_bias(:,NumberInputNeurons+1);
tempH=InputWeight*P;
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);      %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;
clear BiasMatrix
H = 1 ./ (1 + exp(-Gain*tempH));
clear tempH;
OutputWeight=pinv(H') * T';
% Y=(H' * OutputWeight)';
tempH_test=InputWeight*TV.P;
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);      %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
H_test = 1 ./ (1 + exp(-Gain*tempH_test));
TY=(H_test' * OutputWeight)';

if Elm_Type == 1
%%%%%%%%%% Calculate training & testing classification accuracy
    MissClassificationRate_Training=0;
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    Fittness=MissClassificationRate_Training/size(TV.T,2); 
else
    Fittness=sqrt(mse(TV.T - TY));
end