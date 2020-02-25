%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%| SPULTRA main code for 3D XCAT phantom reconstruction
%| Siqi Ye, UM-SJTU Joint Institute
%| 2019-08

clear ; close all; 
addpath('toolbox');
%% setup target geometry and weight
down = 1; % downsample rate
cg = ct_geom('ge2', 'down', down);  
ig = image_geom('nx',420,'dx',500/512,'nz',96,'dz',0.625,'down',down);
% ig.mask = ig.circ > 0; % can be omitted
A = Gcone(cg, ig, 'type', 'sf2');
mm2HU = 1000/0.02;
%% load external parameter
I0 = 1e4; % photon intensit
Sigma = 5; K = 1;
folder = 'spultra/xcat/1e4';
numcluster = 15; % number of classes
gamma0 = 110;
nblock = 12;            % Subset Number
nIter = 4;             % I--Inner Iteration
nOuterIter = 1000;     % T--Outer Iteration
CluInt = 5;           % Clustering Interval
stride = 3; % sliding distance
isCluMap = 0; % 1;          % The flag of caculating cluster mapping 
%load transform: mOmega
load(['transform_ost/block' num2str(numcluster) '_iter2000_gamma110_31l0.mat']);
mOmega = info.mOmega; clear info;
mask_roi = ig.mask(:,:,17:80);
load('data/xtrue_crop17-80.mat');
figure;im('mid3','notick', permute(xtrue,[2 1 3]),[800,1200]),cbar;drawnow
xtrue = xtrue/mm2HU;
xtrue_msk = xtrue(mask_roi);  
%load initial EP image: xrlalm
epbt = 13;
load(['pwls_ep/1e4_xrlalm_l2b' num2str(epbt) '_iter100_os12.mat']);
% im('mid3','notick', permute(xrlalm(:,:,17:80),[2 1 3]),[800,1200]),cbar;drawnow
xrlalm = xrlalm/mm2HU;
xrla_msk = xrlalm(ig.mask); 
xrla = xrlalm .* ig.mask;  
xrla_cat = xrla; 

printm('Loading external sinogram, weight, fbp...');
dir = ['data/' num2str(I0)];
load([dir '/sino_cone.mat']);
load([dir '/Ii_tilde_shp.mat']); % Ii_tilde
load([dir '/maxcurv.mat']); % MaxCurv
load([dir '/kappa.mat']);

%% setup regularizer
ImgSiz =  [ig.nx ig.ny ig.nz];  % image size
PatSiz =  [8 8 8];         % patch size
SldDist = stride * [1 1 1];         % sliding distance
pixmax = inf; %1900;         % Set upper bond for pixel values

Ab = Gblock(A, nblock); clear A

% pre-compute D_R

vLambda = [];
for k = 1:size(mOmega, 3)
    vLambda = cat(1, vLambda, eig(mOmega(:,:,k)' * mOmega(:,:,k)));
end
maxLambda = max(vLambda); clear vLambda;
beta= 4e4;
gamma = 4e-4;
KapPatch = im2colstep(kappa, PatSiz, SldDist); clear kappa;
PatNum = size(KapPatch, 2);
KapPatch = mean(KapPatch,1);
Kappa = col2imstep(single(repmat(KapPatch, prod(PatSiz), 1)), ImgSiz, PatSiz, SldDist);
D_R = 2 * beta * Kappa(ig.mask) * maxLambda;  clear maxLambda Kappa;     
% construct regularizer R(x): add kappa to both regularizer term
R = Reg_OST_Kappa(ig.mask, ImgSiz, PatSiz, SldDist, beta, gamma, KapPatch, mOmega, numcluster, CluInt);
if ~exist([folder '/snapshot/'],'dir') mkdir([folder '/snapshot/']); end
snapname = [folder '/snapshot/epbt' num2str(epbt) '_Sld' num2str(stride) '_nc' num2str(numcluster) '_bt' num2str(beta) '_gm' num2str(gamma)];
info = struct('intensity',I0,'ImgSiz',ImgSiz,'SldDist',SldDist,'beta',beta,'gamma',gamma,...
      'nblock',nblock,'nIter',nIter,'CluInt',CluInt,'pixmax',pixmax,'transform',mOmega,...
      'xrla',[],'vIdx',[],'ClusterMap',[], 'RMSE',[],'SSIM',[],'relE',[],'perc',[],'cost',[]);

%% Recon 
SqrtPixNum = sqrt(sum(mask_roi(:)>0)); % sqrt(pixel numbers in the mask) 
stop_diff_tol = 1e-3/mm2HU;
xroi = xrla(:,:,17:80);
info.RMSE(:,1) = norm(xroi(mask_roi).*mm2HU - xtrue_msk.*mm2HU) / SqrtPixNum; % vector 2-norm 
fprintf('RMSE = %g\n', info.RMSE(:,1));     
RMSE = info.RMSE;
info.SSIM(:,1)= ssim(xroi.*mm2HU, xtrue.*mm2HU);
fprintf('SSIM = %g\n', info.SSIM(:,1)); 
SSIM = info.SSIM; 

Ii = reshape(Ii_tilde,[numel(Ii_tilde),1]);   
for ii=1:nOuterIter
  if mod(ii-1,CluInt)==0
      disp('Do Clustering...')
  end
    xold=xroi;
       
    ln = Ab * xrla_msk;             
    tmp_l = I0 * exp(-ln) + Sigma^2 * ones(size(ln));
    grad_hl = I0 * exp(-ln).* (Ii./tmp_l - ones(size(Ii))); % h'(l)
   
  % Optimum curvature
    threshold = 10; %1e-5;       
    wi = threshold * ones(size(Ii));
    
    % Case 1: index of ln > 0
    index1 = find(ln > 0); 
    if ~isempty(index1) 
    hl = tmp_l - Ii.*log(tmp_l);
    h0 = (I0 + Sigma^2) * ones(size(ln)) - Ii.*log(I0 + Sigma^2);
    curv1 = 2 * (h0(index1) - hl(index1) + grad_hl(index1).*ln(index1))./(ln(index1).^2);
    inter = max(curv1, threshold); 
    wi(index1) = min(inter,MaxCurv(index1)); % eliminates extremly large values caused by too small ln.
    clearvars  h0 curv1 index1 inter;
    end
    
    % Case 2: index of ln == 0
    index2 = find(ln == 0);
    if ~isempty(index2)             
    wi(index2) = MaxCurv(index2);
    end            
    D_A = abs(Ab)'*(wi(:).*(abs(Ab)*ones(length(xrla_msk),1))); 
    wi = reshape(wi,size(Ii_tilde));
    clearvars ggrad_h0 index2;
   ye = ln - grad_hl./wi(:); % (wi(:).^(-1)).* grad_hl;
   ye = reshape(ye,size(Ii_tilde));

  clear tmp_l;

   %% OS-LALM
   fprintf('nOuter = %d,beta = %d, gamma = %d\n', ii,beta, gamma);
   [xrla_msk, cost] = pwls_os_rlalm(xrla_msk, Ab, reshaper(ye,'2d'), reshaper(wi,'2d'), R, D_R, 'denom', D_A,...
          'pixmax', inf, 'alpha', 1.999, 'rho', [], 'niter', nIter, ...
          'chat', 0, 'isave', 'last');    
    xrla = ig.embed(xrla_msk);
    xroi = xrla(:,:,17:80);

figure(1000); title('recon');
% im('mid3','notick', permute(xroi,[2 1 3]),[800/mm2HU 1200/mm2HU]),cbar;drawnow
imshow(xroi(:,:,end/2),[800/mm2HU 1200/mm2HU]);drawnow;

info.RMSE(:,ii+1) = norm(xroi(mask_roi).*mm2HU - xtrue_msk.*mm2HU) / SqrtPixNum; 
% fprintf('outiter = %d, RMSE = %g', ii, info.RMSE(:,ii+1));     xrla_cat = cat(3, xrla_cat, xrla);    
info.SSIM(:,ii+1)= ssim(xroi.*mm2HU, xtrue.*mm2HU);
% fprintf('outiter = %d, SSIM = %g\n', ii, info.SSIM(:,ii+1)); 
% info.xdiff(:,ii)= norm(xrla_msk.*mm2HU - xrla_msk_old.*mm2HU);
fprintf('outiter = %d, RMSE = %g, SSIM = %g, ',...
         ii, info.RMSE(:,ii+1), info.SSIM(:,ii+1));  
if mod(ii,10) == 0
save([snapname '_xrla_iter' num2str(ii) '.mat'],'xrla');
save([snapname '_info.mat'],'info');
end
% xrla_cat = cat(3, xrla_cat, xrla);    

[info.perc(:,ii),info.vIdx] = R.nextOuterIter();
fprintf('perc = %g, ', info.perc(:,ii));          

info.relE(:,ii) =  norm(xroi(mask_roi).*mm2HU - xold(mask_roi).*mm2HU) / SqrtPixNum;
fprintf('relE = %g\n', info.relE(:,ii));   
if info.relE(:,ii) < stop_diff_tol
    break
end

%% Calculate clusterMap 
% % if(isCluMap ~= 0 && R.isClu == R.CluInt)   % don't plot cluster each time, only plot the updates
% %    mPat = im2colstep(single(ig.mask), PatSiz, SldDist);  
% %    PatNum = size(mPat, 2); clear mPat        
% %    info.ClusterMap = ClusterMap(ImgSiz, PatSiz, SldDist, info.vIdx, PatNum, numcluster);
% % %    figure(55), imshow(info.ClusterMap, [1,numcluster]);colorbar  
% %     Cmapint = (info.ClusterMap == 1) .* xrla;
% %    figure(55);
% %    for nc = 2:numcluster
% %        Cmapi = (info.ClusterMap == nc) .* xrla;
% %        Cmapint = cat(2,Cmapint,Cmapi);
% %    end   
% %    Cmap(:,:,isCluMap) = Cmapint(:,:,end/2); clear Cmapint; 
% %    im('mid3','notick', permute(Cmap(:,:,isCluMap),[2 1 3]),[800/mm2HU 1200/mm2HU]),cbar;
% %    isCluMap = isCluMap +1;
% % end  

end

figure name 'RMSE'
plot(info.RMSE,'-+','linewidth',2);grid on;
xlabel('Number of Outer Iteration','fontsize',18)
ylabel('RMSE(HU)','fontsize',18)
title('RMSE');

figure name 'SSIM'
plot(info.SSIM,'-+','linewidth',2);grid on;
xlabel('Number of Outer Iteration','fontsize',18)
ylabel('SSIM','fontsize',18)
title('SSIM');

figure name 'perc'
plot(info.perc,'-+','linewidth',2);grid on;
xlabel('Number of Outer Iteration','fontsize',18)
ylabel('perc','fontsize',18)
title('Percentage');

filename = ['_Sld' num2str(stride) '_epbt' num2str(epbt) '_cInt' num2str(CluInt) 'nc' num2str(numcluster) 'eta' num2str(gamma0)...
    '_bt' num2str(beta) '_gm' num2str(gamma)...
        '_out' num2str(ii) '_os' num2str(nblock) '.mat'];
save([folder '/info' filename],'info');
save([folder '/xrla' filename],'xrla');
 

