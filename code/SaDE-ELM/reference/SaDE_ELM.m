function [TestingAccuracy, TrainingAccuracy, TrainingTime, TY,DE_gbest,DE_gbestval,DE_fitcount,DE_fit_cut,DE_get_flag,ccmhist,pfithist] = SaDE_ELM(TrainingData_File,...
    TestingData_File, Elm_Type, NumberofHiddenNeurons, Max_FES, Lbound, Ubound, NP, Max_Gen, F_par, CR, strategy, numst)

ccmhist=[]; 
pfithist=[];
DE_get_flag = 0; 
DE_fit_cut = Max_FES;
DE_fitcount=0;

aaaa=cell(1,numst); %CR for each strategy
learngen=20;
lpcount=[];
npcount=[];
ns=[];
nf=[];
pfit=ones(1,numst);
ccm = CR*ones(1,numst);

XRmin=-1;
XRmax=1;

REGRESSION=0;
CLASSIFIER=1;
Gain = 1;                                           %  Gain parameter for sigmoid

%%%%%%%%%%% Load training dataset
train_data=load(TrainingData_File);
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data=load(TestingData_File);
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);
NumberofValidationData = round(NumberofTestingData / 2);

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;
end                                                 %   end if of Elm_Type
clear temp_T;
clear temp_T;
VV.P = TV.P(:,1:NumberofValidationData);
VV.T = TV.T(:,1:NumberofValidationData);
TV.P(:,1:NumberofValidationData)=[];
TV.T(:,1:NumberofValidationData)=[];
NumberofTestingData = NumberofTestingData - NumberofValidationData;

D=NumberofHiddenNeurons*(NumberofInputNeurons+1);
tolerance = 0.02;
start_time_validation=cputime;

%-----Initialize population and some arrays-------------------------------
pop = zeros(NP,D); %initialize pop to gain speed
XRRmin=repmat(XRmin,NP,D);
XRRmax=repmat(XRmax,NP,D);
%rand('state',state_no);
pop=XRRmin+(XRRmax-XRRmin).*rand(NP,D);

popold    = zeros(size(pop));     % toggle population
val       = zeros(1,NP);          % create and reset the "cost array"
DE_gbest   = zeros(1,D);           % best population member ever
nfeval    = 0;                    % number of function evaluations

%------Evaluate the best member after initialization----------------------

ibest   = 1;  % start with first population member
[val(1),OutputWeight]  = ELM_X(Elm_Type,pop(ibest,:),P,T,VV,NumberofHiddenNeurons);
DE_gbestval = val(1);                 % best objective function value so far
nfeval  = nfeval + 1;
bestweight = OutputWeight;
for i=2:NP                        % check the remaining members
  [val(i),OutputWeight] = ELM_X(Elm_Type,pop(i,:),P,T,VV,NumberofHiddenNeurons);
  nfeval  = nfeval + 1;
  if (val(i) < DE_gbestval)           % if member is better
     ibest   = i;                 % save its location
     DE_gbestval = val(i);
     bestweight = OutputWeight;
  end   
end
DE_gbest = pop(ibest,:);         % best member of current iteration
ccmhist = [1,ccm];
pfithist = [1,pfit/sum(pfit)];


%------DE-Minimization---------------------------------------------
%------popold is the population which has to compete. It is--------
%------static through one iteration. pop is the newly--------------
%------emerging population.----------------------------------------

pm1 = zeros(NP,D);              % initialize population matrix 1
pm2 = zeros(NP,D);              % initialize population matrix 2
pm3 = zeros(NP,D);              % initialize population matrix 3
pm4 = zeros(NP,D);              % initialize population matrix 4
pm5 = zeros(NP,D);              % initialize population matrix 5
bm  = zeros(NP,D);              % initialize DE_gbestber  matrix
ui  = zeros(NP,D);              % intermediate population of perturbed vectors
mui = zeros(NP,D);              % mask for intermediate population
mpo = zeros(NP,D);              % mask for old population
rot = (0:1:NP-1);               % rotating index array (size NP)
rotd= (0:1:D-1);                % rotating index array (size D)
rt  = zeros(NP);                % another rotating index array
rtd = zeros(D);                 % rotating index array for exponential crossover
a1  = zeros(NP);                % index array
a2  = zeros(NP);                % index array
a3  = zeros(NP);                % index array
a4  = zeros(NP);                % index array
a5  = zeros(NP);                % index array
ind = zeros(4);


iter = 1;
while iter < Max_Gen
    popold = pop;                   % save the old population
    ind = randperm(4);              % index pointer array
    a1  = randperm(NP);             % shuffle locations of vectors
    rt = rem(rot+ind(1),NP);        % rotate indices by ind(1) positions
    a2  = a1(rt+1);                 % rotate vector locations
    rt = rem(rot+ind(2),NP);
    a3  = a2(rt+1);                
    rt = rem(rot+ind(3),NP);
    a4  = a3(rt+1);               
    rt = rem(rot+ind(4),NP);
    a5  = a4(rt+1); 
    
    pm1 = popold(a1,:);             % shuffled population 1
    pm2 = popold(a2,:);             % shuffled population 2
    pm3 = popold(a3,:);             % shuffled population 3
    pm4 = popold(a4,:);             % shuffled population 4
    pm5 = popold(a5,:);             % shuffled population 5
    
    bm = repmat(DE_gbest,NP,1);           
    
    if (iter>=learngen)
        for i=1:numst
            if   ~isempty(aaaa{i}) 
                ccm(i)=median(aaaa{i}(:,1));   
                d_index=find(aaaa{i}(:,2)==aaaa{i}(1,2));
                aaaa{i}(d_index,:)=[];
            else
                ccm(i)=rand;
            end
        end
    end
    
    for i=1:numst
        cc_tmp=[];
        for k=1:NP
            tt=normrnd(ccm(i),0.1);
            while tt>1 | tt<0
                tt=normrnd(ccm(i),0.1);
            end
            cc_tmp=[cc_tmp;tt];
        end
        cc(:,i)=cc_tmp;
    end
    
    % Stochastic universal sampling               %choose strategy
    rr=rand;    
    spacing=1/NP;
    randnums=sort(mod(rr:spacing:1+rr-0.5*spacing,1));  
    
    normfit=pfit/sum(pfit);
    partsum=0;   
    count(1)=0; 
    stpool=[];
    
    for i=1:length(pfit)
        partsum=partsum+normfit(i);
        count(i+1)=length(find(randnums<partsum));
        select(i,1)=count(i+1)-count(i);
        stpool=[stpool;ones(select(i,1),1)*i];
    end
    stpool = stpool(randperm(NP));
    
    for i=1:numst
        atemp=zeros(1,NP);
        aaa{i}=atemp;
        index{i}=[];
        if ~isempty(find(stpool == i))
            index{i} = find(stpool == i);
            atemp(index{i})=1;
            aaa{i}=atemp;
        end
    end
    
    aa=zeros(NP,D);
    for i=1:numst
        aa(index{i},:) = rand(length(index{i}),D) < repmat(cc(index{i},i),1,D);          % all random numbers < CR are 1, 0 otherwise
    end
    mui=aa;
    if (strategy > 1)
        st = strategy-1;		  % binomial crossover
    else
        st = strategy;		  % exponential crossover
        mui=sort(mui');	          % transpose, collect 1's in each column
        for i=1:NP
            n=floor(rand*D);
            if n > 0
                rtd = rem(rotd+n,D);
                mui(:,i) = mui(rtd+1,i); %rotate column i by n
            end
        end
        mui = mui';			  % transpose back
    end
    % jrand
    dd=ceil(D*rand(NP,1));
    for kk=1:NP
        mui(kk,dd(kk))=1; 
    end
    mpo = mui < 0.5;                % inverse mask to mui
    
   for i=1:numst
        %-----------jitter---------
        F=[];
        m=length(index{i});
        F=normrnd(F_par,0.3,m,1);
        F=repmat(F,1,D);
 
        if i==1
            ui(index{i},:) = pm3(index{i},:) + F.*(pm1(index{i},:) - pm2(index{i},:));        % differential variation
            ui(index{i},:) = popold(index{i},:).*mpo(index{i},:) + ui(index{i},:).*mui(index{i},:);     % crossover
        end
        if i==2
            ui(index{i},:) = popold(index{i},:) + F.*(bm(index{i},:)-popold(index{i},:)) + F.*(pm1(index{i},:) - pm2(index{i},:) + pm3(index{i},:) - pm4(index{i},:));       % differential variation
            ui(index{i},:) = popold(index{i},:).*mpo(index{i},:) + ui(index{i},:).*mui(index{i},:);     % crossover
        end
        if i==3
            ui(index{i},:) = pm5(index{i},:) + F.*(pm1(index{i},:) - pm2(index{i},:) + pm3(index{i},:) - pm4(index{i},:));       % differential variation
            ui(index{i},:) = popold(index{i},:).*mpo(index{i},:) + ui(index{i},:).*mui(index{i},:);     % crossover
        end
        if i==4     
            ui(index{i},:) = popold(index{i},:) + rand.*(pm5(index{i},:)-popold(index{i},:)) + F.*(pm1(index{i},:) - pm2(index{i},:));  
        end
    end    
    
    for i=1:NP
        outbind=find(ui(i,:) < Lbound)
        if size(outbind,2)~=0
            %                 % Periodica
            %                 ui(i,outbind)=2*XRmin(outbind)-ui(i,outbind);
            % Random
            ui(i,outbind)=XRRmin(outbind)+(XRRmax(outbind)-XRRmin(outbind)).*rand(1,size(outbind,2));
            %                 % Fixed
            %                 ui(i,outbind)=XRmin(outbind);
        end            
        outbind=find(ui(i,:) > Ubound)
        if size(outbind,2)~=0
            %                 % Periodica
            %                 ui(i,outbind)=2*XRmax(outbind)-ui(i,outbind);
            % Random
            ui(i,outbind)=XRRmin(outbind)+(XRRmax(outbind)-XRRmin(outbind)).*rand(1,size(outbind,2));
            %                 % Fixed
            %                 ui(i,outbind)=XRmax(outbind);
        end
    end
    lpcount=zeros(1,numst); 
    npcount=zeros(1,numst);
    for i=1:NP
        [tempval,OutputWeight] = ELM_X(Elm_Type,ui(i,:),P,T,VV,NumberofHiddenNeurons);
        %tempval = feval(fname,ui(i,:),varargin{:});   % check cost of competitor
        nfeval  = nfeval + 1;
        if (tempval <= val(i))  % if competitor is better than value in "cost array"
            pop(i,:) = ui(i,:);  % replace old vector with new one (for new iteration)
            val(i)   = tempval;  % save value in "cost array"
            tlpcount=zeros(1,numst);
            for j=1:numst
                temp=aaa{j};
                tlpcount(j)=temp(i);
                if tlpcount(j)==1
                    aaaa{j}=[aaaa{j};cc(i,j) iter]  
                end
            end
            lpcount=[lpcount;tlpcount];
            %----we update DE_gbestval only in case of success to save time-----------
            if DE_gbestval-tempval>tolerance*DE_gbestval
           % if (tempval <= DE_gbestval)     % if competitor better than the best one ever
                DE_gbestval = tempval;      % new best value
                DE_gbest = ui(i,:);      % new best parameter vector ever
                bestweight = OutputWeight;
                %if DE_gbestval <= VTR && DE_get_flag == 0
                elseif abs(tempval-DE_gbestval)<tolerance*DE_gbestval    % if competitor better than the best one ever
                    if norm(OutputWeight,2)<norm(bestweight,2)
                        DE_gbestval = tempval;      % new best value
                        DE_gbest = ui(i,:);      % new best parameter vector ever
                        bestweight = OutputWeight;
                    DE_fit_cut=nfeval;
                    DE_get_flag=1;
%                     DE_fitcount = nfeval;
%                     return;
                end
            end
        else
            tnpcount=zeros(1,numst);
            for j=1:numst
                temp=aaa{j};
                tnpcount(j)=temp(i);
            end
            npcount=[npcount;tnpcount];
        end
        
        if nfeval+1 > Max_FES
            DE_fitcount = Max_FES;
            pfithist = [pfithist;[iter+2,pfit/sum(pfit)]]
            ccmhist = [ccmhist;[iter+2,ccm]]
            return;
        end
        %     end
    end %---end for imember=1:NP
    pfithist = [pfithist;[iter+2,pfit/sum(pfit)]]
    ccmhist = [ccmhist;[iter+2,ccm]];    
    ns=[ns;sum(lpcount,1)]
    nf=[nf;sum(npcount,1)]    
    
    if iter >= learngen,
        %         ww=repmat((1:learngen)',1,numst);
        %         ns=ww.*ns;
        %         nf=ww.*nf;
        for i=1:numst            
            if (sum(ns(:,i))+sum(nf(:,i))) == 0
                pfit(i) = 0.01;
            else
                pfit(i) = sum(ns(:,i))/(sum(ns(:,i))+ sum(nf(:,i))) + 0.01;
            end
        end
        if ~isempty(ns), ns(1,:)=[];   end
        if ~isempty(nf), nf(1,:)=[];   end
    end 
    iter = iter + 1;
end
%---end while ((iter < Max_Gen) ...
end_time_validation=cputime;
TrainingTime=end_time_validation-start_time_validation

%%%%%%%%%%%%% Testing the performance of the best population
Output_weight = mean(abs(OutputWeight))
NumberInputNeurons=size(P, 1);
NumberofTrainingData=size(P, 2);
NumberofTestingData=size(TV.P, 2);
Gain=1;
temp_weight_bias=reshape(DE_gbest, NumberofHiddenNeurons, NumberInputNeurons+1);
InputWeight=temp_weight_bias(:, 1:NumberInputNeurons);
BiasofHiddenNeurons=temp_weight_bias(:,NumberInputNeurons+1);
tempH=InputWeight*P;
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);      %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;
clear BiasMatrix
H = 1 ./ (1 + exp(-Gain*tempH));
clear tempH;
% OutputWeight=pinv(H') * T';
Y=(H' * bestweight)';
tempH_test=InputWeight*TV.P;
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);      %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
H_test = 1 ./ (1 + exp(-Gain*tempH_test));
TY=(H_test' * bestweight)';
if Elm_Type == 0
    TrainingAccuracy=sqrt(mse(T - Y))
    TestingAccuracy=sqrt(mse(TV.T - TY))            %   Calculate testing accuracy (RMSE) for regression case
end

if Elm_Type == 1
%%%%%%%%%% Calculate training & testing classification accuracy
    MissClassificationRate_Testing=0;
    MissClassificationRate_Training=0;
    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2)
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)
end
