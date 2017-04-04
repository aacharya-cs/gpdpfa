function [idx,idxh] = holdoutentries(A,p,mode)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% input:
%% A: oroginal matrix
%% p: fraction of ones to be held out
%% mode: 'equal'  for holding out equal number of links and non-links
%%       'nequal' for holding out 100*p % of links and non-links from the entire matrix
%%       'other'  for holding out 100*p % of the positive entries only
%%       'column-stratified'  for holding out 100*p % of the positive entries only with equal number of entries from each column
%% output:
%% idx:  indices of links from the non-held-out set
%% idxh: indices of links and non-links from the held-out set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(p==0)
    %% don't hold out anything
    [nn,pp,bb] = find(A);
    idx        = sub2ind(size(A),nn,pp);      %% indices of ones
    idxh       = [];
elseif(p>=1.0)
    error('set p<1.0');
else
    [N,D]  = size(A);
    [nn,pp,bb] = find(A);
    idx = [1:N*D]';
    idxones  = sub2ind(size(A),nn,pp);                 %% indices of ones
    if strcmp(mode,'equal')                            %% mode: selects equal number of links and nonlinks
        totmis = floor(p*length(find(A)));
        idxzeros = setdiff(idx,idxones);               %% indices of zeros
        temp1 = SelRandomVec(length(idxones),totmis);  %% ones
        temp2 = SelRandomVec(length(idxzeros),totmis); %% zeros
        idxh  = [idxones(temp1)' idxzeros(temp2)']';
        idx   = setdiff(idxones,idxh);
    elseif (strcmp(mode,'nequal'))
        totmis = floor(p*N*D);
        idxh = SelRandomVec(length(idx),totmis);       %% links and non-links
        idx  = setdiff(idxones,idxh);
    elseif (strcmp(mode,'column-stratified'))
        [ii,jj,~]   = ind2sub(size(A), idx);
        idxh   = [];
        T      = size(A,2);
        for t=1:T
            percolones = jj(find(ii==t));
            percol     = floor(p*length(percolones));
            jjind      = percolones(SelRandomVec(length(percolones),percol));
            iiind      = t*ones(length(jjind),1);
            idxh       = [idxh sub2ind(size(A),iiind,jjind);];
            %% only positive entries
        end
        idx    = setdiff(idxones,idxh);
    else
        %%do nothing for now
    end
end

end
