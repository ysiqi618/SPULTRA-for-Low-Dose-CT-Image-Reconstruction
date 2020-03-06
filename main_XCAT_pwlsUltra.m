%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clear ; close all;
addpath('toolbox');
%% setup target geometry and weight
down = 1; % downsample rate
cg = ct_geom('ge2', 'down', down);
ig = image_geom('nx',420,'dx',500/512,'nz',96,'dz',0.625,'down',down);
% ig.mask = ig.circ > 0;
A = Gcone(cg, ig, 'type', 'sf2','nthread', jf('ncore')*2);
mm2HU = 1000/0.02;
%% load external parameter
I0 = 1e4; % photon intensity
epbt = 13; % beta of EP 2^epbt
folder = 'pwls_ultra/xcat/1e4';
load(['data/pwls_ep/1e4_xrlalm_l2b' num2str(epbt) '_iter100_os12.mat']);
xrlalm = xrlalm/mm2HU;
ImgSiz = [ig.nx ig.ny ig.nz]; % image size
PatSiz = [8 8 8];         % patch size
SldDist = 3 * [1 1 1];        % sliding distance

nblock = 12;            % Subset Number
numcluster = 15;
nOuterIter = 800;     % Outer Iteration
nIter = 4;             % Inner Iteration
CluInt = 5;           % Clustering Interval
isCluMap = 1;          % The flag of caculating cluster mapping
pixmax = inf;          % Set upper bond for pixel values
gamma0 = 110; % training eta
% load transform: mOmega
% load('885_slice101_154_block15_SldDist2_iter1000_gamma150.mat');
load(['transform_ost/block' num2str(numcluster) '_iter2000_gamma110_31l0.mat']);
mOmega = info.mOmega; clear info;
% load ground truth image: xtrue
load('data/xtrue_crop17-80.mat');
% xtrue = xtrue/mm2HU;
% load measurements and initial data
dir = ['data/' num2str(I0)];
printm('Loading sinogram, weighting, xfdk...');
load([dir '/sino_cone.mat']);
load([dir '/wi.mat']);
% load([dir '/xfdk.mat']);
% figure; imshow(xfdk(:,:,end/2), [800 1200]);

%% setup edge-preserving regularizer
xrla_msk = xrlalm(ig.mask);  % initial EP image
%xrla = xfdk .* ig.mask;     % initial FDK image
%xrla_msk = xfdk(ig.mask);

% set up ROI
start_slice = 17; end_slice = 80;
xroi = xrlalm(:,:,start_slice:end_slice); clear xrlalm
mask_roi = ig.mask(:,:,start_slice:end_slice);
% roi = ig.mask; roi(:,:,1:start_slice-1) = 0; roi(:,:,end_slice+1:end) = 0;
% roi = roi(ig.mask);



printm('Pre-calculating denominator: D_A...');
% denom = A' * col(reshape(sum(A'), size(wi)) .* wi);
% denom=  A'*(wi(:).*(A*ones(size(xrla_msk,1),1)));
load([dir '/denom.mat']);

Ab = Gblock(A, nblock); clear A

% pre-compute D_R
numBlock = size(mOmega, 3);
vLambda = [];
for k = 1:numBlock
    vLambda = cat(1, vLambda, eig(mOmega(:,:,k)' * mOmega(:,:,k)));
end
maxLambda = max(vLambda); clear vLambda;


beta= 4e4;
gamma = 4e-4;
 
load([dir '/kappa.mat']);
KapPatch = im2colstep(kappa, PatSiz, SldDist); clear kappa;
PatNum = size(KapPatch, 2);
KapPatch = mean(KapPatch,1);
Kappa = col2imstep(single(repmat(KapPatch, prod(PatSiz), 1)), ImgSiz, PatSiz, SldDist);
D_R = 2 * beta * Kappa(ig.mask) * maxLambda;  clear maxLambda Kappa;     
% construct regularizer R(x): add kappa to both regularizer term
R = Reg_OST_Kappa(ig.mask, ImgSiz, PatSiz, SldDist, beta, gamma, KapPatch, mOmega, numBlock, CluInt);
            

fprintf('beta = %.1e, gamma = %g: \n\n', beta, gamma);
if ~exist([folder '/snapshot/'],'dir') mkdir([folder '/snapshot/']); end
snapname = [ folder '/snapshot/epbt' num2str(epbt) '_nc' num2str(numcluster) '_bt' num2str(beta) '_gm' num2str(gamma)];

info = struct('intensity',I0,'ImgSiz',ImgSiz,'PatSiz',PatSiz,'SldDist',SldDist,'beta',beta,'gamma',gamma,...
    'nblock',nblock,'nIter',nIter,'CluInt',CluInt,'pixmax',pixmax,'transform',mOmega,...
    'xrla',[],'vIdx',[],'ClusterMap',[],'RMSE',[],'SSIM',[],'relE',[],'perc',[],'cost',[]);
%% Recon
SqrtPixNum = sqrt(sum(mask_roi(:)>0));
stop_diff_tol = 1e-3/mm2HU; 
iterate_fig = figure(55);
idx_old = ones([1,PatNum],'single');

for ii=1:nOuterIter
    figure(iterate_fig); drawnow;
    xold = xroi;
    AAA(1,ii) = norm(xroi(mask_roi).*mm2HU - xtrue(mask_roi)) / SqrtPixNum;
    info.RMSE = AAA(1,:);
    AAA(2,ii) = ssim(xroi.*mm2HU, xtrue);
    fprintf('RMSE = %g, SSIM = %g\n',  AAA(1,ii), AAA(2,ii));
    info.SSIM = AAA(2,:);
    
    fprintf('Iteration = %d:\n', ii);    
    [xrla_msk, cost] = pwls_os_rlalm(xrla_msk, Ab, reshaper(sino,'2d'), reshaper(wi,'2d'), R, D_R, 'denom', denom,...
          'pixmax', inf, 'alpha', 1.999, 'rho', [], 'niter', nIter, ...
          'chat', 0, 'isave', 'last'); % , 'nblock', nblock);    
    %      [xrla_msk, cost] = pwls_sqs_os_mom(xrla_msk, Ab, reshaper(sino,'2d'),...
    %                    reshaper(wi,'2d'), R, denom, D_R, nIter, pixmax, 2, 0);
    
    [info.perc(:,ii),info.vIdx] = R.nextOuterIter();
    fprintf('perc = %g\n', info.perc(:,ii));

    idx_diff = idx_old - info.vIdx;
    fprintf('Idx Change Perc = %g\n', nnz(idx_diff)/PatNum);
    idx_old = info.vIdx;
    
%     info.cost(:,ii) = cost;
    xrla = ig.embed(xrla_msk); xroi = xrla(:,:,start_slice:end_slice);
    info.relE(:,ii) =  norm(xroi(mask_roi).*mm2HU - xold(mask_roi).*mm2HU) / SqrtPixNum;
    fprintf('relE = %g\n', info.relE(:,ii));
    if info.relE(:,ii) < stop_diff_tol
        break
    end
    figure(120), imshow(xroi(:,:,end/2).*mm2HU, [800 1200]); drawnow;
     if mod(ii,10) == 0
    save([snapname '_xrla_iter' num2str(ii) '.mat'],'xrla');
    save([snapname '_infoOutiter.mat'],'info');
    end

end

figure name 'RMSE'
plot(info.RMSE,'-+')
xlabel('Number of Outer Iteration','fontsize',18)
ylabel('RMSE(HU)','fontsize',18)
legend('PWLS-OST')

figure name 'SSIM'
plot(info.SSIM,'-+')
xlabel('Number of Outer Iteration','fontsize',18)
ylabel('SSIM','fontsize',18)
legend('PWLS-OST')

filename = ['_epbt' num2str(epbt) '_cInt' num2str(CluInt) 'nc' num2str(numcluster) 'eta' num2str(gamma0)...
    '_bt' num2str(beta) '_gm' num2str(gamma)...
        '_out' num2str(ii) '_os' num2str(nblock) '.mat'];
save([folder '/info' filename],'info');
save([folder '/xrla' filename],'xrla');

