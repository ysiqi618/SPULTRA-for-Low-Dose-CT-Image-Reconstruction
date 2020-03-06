%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clear; close all; 
addpath(genpath('toolbox'));


%% generate noisy sinogram and statistical weighting
I0 = 1e4;
down = 1;% downsample rate
cg = ct_geom('ge2'); 
%  cg.na = 246; % sparse view

dir = ['data/' num2str(I0)];
% dir = ['~/myproject/PL+ST/3Dxcat/data/' num2str(cg.na)];

load('data/phantom_crop1-96.mat');  % 
mm2HU = 1000 / 0.02; 
% phantom = phantom/cm2HU;
figure;im('mid3','notick', permute(phantom, [2 1 3]),[800,1200]),cbar
fprintf('generating noiseless sino...\n');
ig_hi = image_geom('nx',840,'dx',500/1024,'nz',96,'dz',0.625,'down',down);
A_hi = Gcone(cg, ig_hi, 'type', 'sf2');  
sino_true = A_hi * phantom;  clear A_hi;

fprintf('adding noise...\n');
yi = poisson(I0 * exp(-sino_true ./ mm2HU), 0, 'factor', 0.4);
sigma = 5;
var = sigma^2 .* ones(size(yi),'single');
ye = sqrt(var).* randn(size(yi)); % Gaussian white noise
k = 1;
zi = k * yi + ye; clear yi; 
Ii_tilde = zi/k + var/(k^2); % Ii_tilde = Ii + simga^2    mm
save([dir '/Ii_tilde_shp.mat'], 'Ii_tilde'); 
error = 1/1e5;
no_positive = sum(zi(:) < error)/length(zi(:))*100; fprintf('no_positive = %f', no_positive);
zi = max(zi, error);   


sino = -log(zi ./(k*I0)); % * mm2HU; % HU

wi = (zi.^2)./(k*zi + var); clear zi; % weighting
save([dir '/wi.mat'], 'wi');    % mm
save([dir '/sino_cone.mat'], 'sino'); 

Curv = (ones(size(Ii_tilde)) - Ii_tilde * sigma^2 / (I0 + sigma^2)^2) * I0;
threshold = 1e-5; % 10
MaxCurv = max(Curv,threshold);
save([dir '/maxcurv.mat'], 'MaxCurv');  

%% setup target geometry and fbp  (HU)
ig = image_geom('nx',420,'dx',500/512,'nz',96,'dz',0.625,'down',down);
% ig.mask = ig.circ > 0;
A = Gcone(cg, ig, 'type', 'sf2');
fprintf('fdk...\n');
xfdk = feldkamp(cg,ig,sino,'window','hanning,0.35','w1cyl',1,'extrapolate_t',round(1.3*cg.nt/2));
xfdk = max(xfdk , 0); % HU;
save([dir '/xfdk.mat'], 'xfdk');
%% setup kappa
fprintf('calculating kappa...\n');
kappa = sqrt( div0(A' * wi, A' * ones(size(wi))) );
kappa = max(kappa, 0.01*max(col(kappa)));
save([dir '/kappa.mat'], 'kappa');
figure(20), im('mid3',permute(kappa,[1 2 3]))
% %% setup D_A
printm('Pre-calculating denominator D_A...');
denom = abs(A)' * col(reshape(sum(abs(A)'), size(wi)) .* wi); % no abs also fine
save([dir '/denom.mat'], 'denom');
