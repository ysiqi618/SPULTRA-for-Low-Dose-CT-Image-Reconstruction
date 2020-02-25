%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clear ; close all;
addpath(genpath('toolbox'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Initialization  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/phantom_crop154.mat');

PatSiz = [8 8 8];  % patch size
SldDist = 2 * [1 1 1];  % sliding distance

lambada0  = 31;   % set it large enough to guarantee well-conditioned
iter = 2000;  % iteration
gamma = 110;
numBlock = 15;     % cluster number
nz = 34;     % number of training slices

for ii=1:nz
  image(:,:,ii) = downsample2(phantom(:,:,ii+120), 2);
end
clear phantom;
ImgSiz = size(image);

fprintf('extracting patches...\n')
%   patch = volume2patch(image, PatSiz, SldDist); clear image;
patch = im2colstep(single(image), PatSiz, SldDist); clear image;
patch = double(patch);
PatNum = size(patch, 2);
fprintf('Length of training set: %d\n', PatNum);

% fprintf('K-mean Initializing...');
% [IDX, ~] = kmeans(patch',numBlock);   % K-mean Initialization (slow for 3D!)
IDX = randi(numBlock,PatNum,1);       % Random Initialization

TransWidth = prod(PatSiz);

D1 = dctmtx(PatSiz(1));
D2 = dctmtx(PatSiz(2));
D3 = dctmtx(PatSiz(3));
D = kron(kron(D1, D2), D3);  % 3D DCT Initialization. Be careful of the order of D1 D2 D3!
clear D1 D2 D3

mOmega = zeros(TransWidth,TransWidth,numBlock, 'double');% must be 'double'!
for i = 1:numBlock
  mOmega(:,:,i) = D;
end
clear D

perc = zeros(iter,numBlock,'single'); % sparsity (percentage)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j = 1 : iter
  fprintf('iteration = %d:\n', j);
  for k = 1 : numBlock
    patch_k = patch(:, IDX == k); % this way is faster
    lambada_k = lambada0 * norm(patch_k, 'fro') ^ 2;% lambda_{k} update
    sparseCode = mOmega(:,:,k) * patch_k;
    % hard-thresholding
    sparseCode = sparseCode.*(abs(sparseCode) > gamma);
    perc(j,k) = nnz(sparseCode) / numel(sparseCode)* 100;
    %         fprintf('sparsity(%d) = %g\n', k, perc(j,k));
    
    % transform update
    if (size(patch_k,2) > 0) % if patch_k is empty, transform will be unchange
      [U,S,V]=svd((patch_k*patch_k') + (lambada_k*eye(TransWidth)));
      LL2=(inv(U*(S^(1/2))*V')); clear U S V
      [Q1,Si,R]=svd(LL2*patch_k*sparseCode');
      sig=diag(Si); clear Si
      gamm=(1/2)*(sig + (sqrt((sig.^2) + 2*lambada_k)));
      B=R*(diag(gamm))*Q1';
      mOmega(:,:,k)=B*(LL2);
    else
      fprintf('patch %g is empty\n', k);
    end
  end
  clear patch_k sparseCode LL2 Q1 sig R B
  fprintf('Cond Number(%d) = %g\n', numBlock, cond(mOmega(:,:,numBlock)));
  fprintf('sparsity(%d) = %g\n', numBlock, perc(j,numBlock));
  error = zeros(numBlock, PatNum, 'double');
  %%%%%%%%% clustering measure %%%%%%%
  for k = 1 : numBlock
    a1 = mOmega(:,:,k) * patch;
    % hard-thresholding
    a0 = a1 .* (abs(a1) > gamma);
    error(k, :) = sum((a1-a0).^2,'double') + gamma^2 * sum(abs(a0) > 0) ...
      + lambada0 * sum((patch).^2,'double') *...
      (sum(col(mOmega(:,:,k).^2),'double') - log(abs(det(mOmega(:,:,k))))) ;
  end
  % clustering
  [~, IDX] = min(error, [] ,1); clear  error a1 a0
end
%%%%%%%%%%%%%%%%%%%% check cluster-mapping %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CluMap = ClusterMap(ImgSiz, PatSiz, SldDist, IDX, PatNum, numBlock);
%  figure(55), imshow(CluMap(:,:,end/2), [1,numBlock]);colorbar
%  figure(55), im('mid3',permute(CluMap,[2 1 3]), [1,numBlock])
%  figure(55), im('mid3',permute((CluMap == numBlock).* image,[2 1 3]), [800,1200])

%%%%%%%%%%%%%%%%%%%%%% check convergency %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure();
for k = 1 : numBlock
  handles(k) = plot(perc(:,k));hold on;
  lables{k} = sprintf('cluster %d',k);
end
legend(handles,lables{:});
xlabel('Number of Iteration','fontsize',18)
ylabel('Sparity ( % )','fontsize',18)
%%%%%%%%%%%%%%%%%%%%%% check condition number %%%%%%%%%%%%%%%%%%%%%%%%%%%%
condTransform = zeros(numBlock, 1);
for k = 1 : numBlock
  condTransform(k, 1) = cond(mOmega(:, :, k));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

info = struct('lambada0',lambada0,'ImgSiz',ImgSiz,'SldDist',SldDist,'gamma',gamma,...
 'numBlock',numBlock,'iter',iter,'mOmega',mOmega,'IDX',IDX ,'perc',perc);
% save(['transform_ost/block' num2str(numBlock) '_iter' num2str(iter) ...
    '_gamma' num2str(gamma) '_31l0.mat'],'info');

transform = mOmega(1:256,1:64,5);
% 
ka = 16-1;
kb = 16-1;
for i=1:size(transform,1)
    transform(i,:)=transform(i,:)-min(transform(i,:));
    if(max(transform(i,:))>0)
      transform(i,:)=transform(i,:)/(max(transform(i,:)));
    end
end
   
jy=1;cc=1;
Ta=(max(max(transform)))*zeros((8+jy)*ka + 8,(8+jy)*kb + 8);
for i=1:8+jy:(ka*(8+jy))+1
    for j=1:8+jy:(kb*(8+jy))+1
       Ta(i:i+7,j:j+7)=reshape((transform(cc,:))',8,8);
       cc=cc+1;
    end
end
figure();imagesc(Ta);colormap('Gray');axis off;axis image; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% perc
figure;
mark = {'-','--','-.','-r','--r','-.r','-k','--k','-.k','-m','--m','-.m','-g','--g','-.g'};
for i=1:numBlock
plot(info.perc(:,i),mark{i});
hold on;
end
