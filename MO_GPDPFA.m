function [] = MO_GPDPFA()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Code for AISTATS-15 submission of GP-DPFA; uses only .m scripts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if 0
    %Xorg     = round(5*rand(100,10));
    Xorg     = [4,5,4,1,0,4,3,4,0,6,3,3,4,0,2,6,3,3,5,4,5,3,1,4,4,1,5,5,3,4,2,5,2,2,3,4,2,1,3,2,2,1,1,1,1,3,0,0,1,0,1,1,0,0,3,1,0,3,2,2,0,1,1,1,0,1,0,1,0,0,0,2,1,0,0,0,1,1,0,2,3,3,1,1,2,1,1,1,1,2,4,2,0,0,0,1,4,0,0,0,1,0,0,0,0,0,1,0,0,1,0,1];
end
if 0
    Xorg = load('STU/STOU.mat');
    Xorg = Xorg.STOU.WCmatrix;
    Xorg(find(Xorg<=7)) = 0;
    temp = sum(Xorg,2);
    length(find(temp>0))
    max(max(Xorg))
end
if 1
    Xorg = load('GhoshPapers.mat');
    Xorg = full(Xorg.B);
    Xorg(find(Xorg<=2)) = 0;
    Xorg = Xorg';
    VocabIndex = load('VocabIndex.mat');
    VocabIndex = VocabIndex.untitled;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% model parameters
K = 100; c = 1; azero = 1; bzero = 1.0; ezero=1.0; fzero=1.0; gammazero = 10.0; hzero = 1.0; etazero = 0.01;
%% set K=V=1 for GPAR
%% Gibbs sampling specific parameters
burnin  = 4000; collection = 4000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
futureT  = 0;   %% number of time-stamps over which every observation is held-out
p        = 0.1; %% fraction of observation held-out from all other time-stamps
Xfuture  = Xorg(:,end-futureT+1:end);
X        = Xorg(:,1:end-futureT);
[V,T]    = size(X);
indexset = [1:(T+futureT)];
%% hold-out entries from this matrix for prediction of missing entries
[idx,idxh]  = holdoutentries(X,p,'column-stratified');
[ii,jj,~]   = ind2sub(size(X), idx);
[iih,jjh,~] = ind2sub(size(X), idxh);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% for the proposed model
Phi       = rand(V,K);
Phi       = Phi./repmat(sum(Phi,2),1,K);
Theta     = zeros(T,K)+1/K;
lambdak   = ones(1,K);
thetazero = hzero*ones(T,K);
if(K==1 && V==1)
    Samples   = zeros(T+futureT,collection);
end
ssTheta   = zeros(T,K);
ssPhi     = zeros(V,K);
ssLambdak = zeros(1,K);
%% for future observations
ctprime      = zeros(1,futureT);
thetaktprime = zeros(futureT,K);
nprime       = zeros(V,futureT);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for iter=1:burnin + collection
    iter
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% sampling of latent counts
    %% n_w_dot_k: K \times V;
    %% n_dot_t_k: K \times T;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    n_w_dot_k = sparse(zeros(K,V));
    n_dot_t_k = sparse(zeros(K,T));
    for tw=1:length(idx)
        pmf                 = Phi(ii(tw),:).*lambdak.*Theta(jj(tw),:);
        ntw_k               = multrnd_histc(X(tw),full(pmf));
        n_w_dot_k(:,ii(tw)) = n_w_dot_k(:,ii(tw)) + ntw_k;
        n_dot_t_k(:,jj(tw)) = n_dot_t_k(:,jj(tw)) + ntw_k;
    end
    sumTheta = (sum(Theta,2))';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% sample c's first; T-specific variables
    c_t = gamrnd(ezero + [sum(sum(thetazero)),sumTheta(1:end-1)],1./(fzero+sumTheta));
    L_t = zeros(K,T+1); 
    p_t = zeros(K,T+1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% backward sampling; only the L's which are K and T-specific variables
    for t=T:-1:2
        p_t(:,t)     = (1 - c_t(t)./max(c_t(t) + max(lambdak',eps) - log(max(1-p_t(:,t+1),eps)),eps));
        [~,L_t(:,t)] = CRT(n_dot_t_k(:,t)+L_t(:,t+1),Theta(t-1,:));
    end
    [~,L_t(:,1)] = CRT(n_dot_t_k(:,1)+L_t(:,2),gammazero/K);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% forward sampling; only the theta's which are K and T-specific variables
    c_t0       = 1;
    thetazero  = gamrnd(hzero/K+ L_t(:,1),1./max(c_t0+max(lambdak',eps)-log(max(1-p_t(:,1),eps)),eps)) ;
    Theta(1,:) = gamrnd(thetazero+n_dot_t_k(:,1)+L_t(:,2),1./max(c_t(1)+lambdak'-log(max(1-p_t(:,2),eps)),eps));
    for t=2:T
        param1     = Theta(t-1,:)+(n_dot_t_k(:,t)+L_t(:,t+1))';
        param2     = (1./(max(c_t(t)+lambdak'-log(max(1-p_t(:,t+1),eps)),eps)))';
        Theta(t,:) = gamrnd(param1,param2);
    end
    ell     = CRT_sum(sum(n_dot_t_k,2),gammazero/K);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% sample k-specific variables
    Phi     = (dirrnd(n_w_dot_k + etazero))';
    param1  = (sum(n_dot_t_k,2)+gammazero/K)';
    param2  = 1./(c+sum(Theta,1));
    lambdak = gamrnd(param1,param2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% sample global variables
    gammazero = gamrnd(ell+ezero, 1/(fzero-sum(log(c./(c+sum(Theta,1))))));
    c         = gamrnd(ezero + gammazero,1./(fzero+sum(lambdak)));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% generate future data; plot results for GPAR
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(iter>burnin)
        %% update sufficient statistics
        ssTheta   = ssTheta + Theta;
        ssPhi     = ssPhi + Phi;
        ssLambdak = ssLambdak + lambdak;
        for tprime=1:futureT
            %% sample c_t
            ctprime(tprime) = c_t(end); %%gamrnd(ezero,1/fzero);
            %% sample theta_kt
            if(tprime>1)
                thetaktprime(tprime,:) = gamrnd(thetaktprime(tprime-1,:),1/ctprime(tprime)); %%1.0./(ctprime(tprime)*ones(K,1)));
            else
                thetaktprime(tprime,:) = gamrnd(Theta(T,:),1/ctprime(tprime)); %%1.0./(ctprime(tprime)*ones(K,1)));
            end
            %% sample n's
            nprime(:,tprime) = poissrnd((repmat(lambdak,V,1).*Phi)*(thetaktprime(tprime,:))');
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end        
end

if(K==1 && V==1) %% for the GPAR model and its baseline
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% for proposed model
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    temp     = [Theta' thetaktprime']';
    currrate = [temp*diag(lambdak)]'./[c_t ctprime];
    Samples(:,iter-burnin) = currrate;
    temp = (mean(Samples,2))';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if 1 %(mod(iter,50)==0)
        %figure(1);
        %plot(indexset,temp,'r.-',indexset,tempbase,'k.-',indexset, Xtemp, 'b.-', indexset, currrate, 'g.-', 'linewidth', 2);
        plot(indexset,temp,'r.-',indexset,tempbase,'k.-',indexset, Xorg, 'b.-', 'linewidth', 5);
        legend('estimated rate','estimated rate (baseline)','original count'); grid on;
        axis([1 T 0 10]);
        xlabel('year index');
        ylabel('count');
        set(gca,'FontSize',40,'fontWeight','bold'),
        set(findall(gcf,'type','text'),'FontSize',40,'fontWeight','bold'),
        %                 figure(2);
        %                 %plot(indexset,tempbase,'r.-',indexset, Xtemp, 'b.-', indexset, currratebase, 'g.-', 'linewidth', 2);
        %                 plot(indexset,tempbase,'r.-',indexset, Xtemp, 'b.-', 'linewidth', 2);
        %                 legend('estimated rate','original count','current sample'); grid on;
        %                 axis([1 T 0 10]);
        %                 xlabel('year index');
        %                 ylabel('count');
        drawnow;
    end
end

if(K==1 && V==1) %% do nothing for GPAR as of now
else
    %% calculate precision and recall after estimation
    Msz   = 50;
    Xpred = round((ssPhi/collection)*diag(ssLambdak/collection)*(ssTheta'/collection));
    for t=1:T
        ind          = find(iih==t);
        orgtestind   = jjh(ind);
        [~,orgind]   = sort(X(:,t),'descend');
        [~,predind]  = sort(Xpred(:,t),'descend');
        precision(t) = length(intersect(orgind(1:Msz),predind(1:Msz)))/Msz;
        recall(t)    = length(intersect(orgtestind,predind(1:Msz)))/Msz;
    end        
    mean(precision)
    mean(recall)
end

%% plot for the DBLP data
% subplot();
% for t=1:T
%     temp = lambdak*Theta(t,:);
%     subplot(T,1,(t-1)*3+3); plot(temp/sum(temp),'r.-');title('relative strength of factors');
%     xlabel('topic index');
%     ylabel('topic strength');
%     set(gca,'FontSize',10,'fontWeight','bold'),
%     set(findall(gcf,'type','text'),'FontSize',10,'fontWeight','bold'),
%     temp = lambdakbase*Thetabase(t,:);
%     subplot(T,2,(t-1)*3+3); plot(temp/sum(temp),'b.-');title('relative strength of factors');
%     xlabel('topic index');
%     ylabel('topic strength');
%     set(gca,'FontSize',10,'fontWeight','bold'),
%     set(findall(gcf,'type','text'),'FontSize',10,'fontWeight','bold'),
% end

end

