t0 = clock;
% Parameter Setting
N=3;
M=224;
L=10000;

% Algorithm
load data30
X = reshape(X3D,L,M)'; % 3D to 2D

%Generate missing data(Revise Here)
Y = X;
pixelcut = [1:1702 , 1767:2500 , 2710:2722 , 2800:3500 , 3520 , 3545:3596 , 3600 , 3710 , 3800:10000]; 
bandcut = [1:220 , 223:224];                                                                           
Y(bandcut , pixelcut) =0;

% Subset 
Y_Omega = X;
Y_Omega(: , pixelcut) = [];
 
% Obtain A by HyperCSI
[~ , L] = size(Y_Omega);        
[A, timeHyperCSI,a] = HyperCSI(Y_Omega,N);

% Subset
A_Omega = A;    
A_Omega(bandcut,:) = [];   
Y_Omega2 = X;
Y_Omega2(bandcut , :) = [];

% PCA Dimension Reduction of Y_Omega2
[Y_re , L] = PCA(Y_Omega2,N);

% Obtain Hyperplane from A_Omega
[h_hat , b_hat , alpha_hat] = Hyperplane(A_Omega, N);

% Obtain S
S = ( h_hat*ones(1,L)- b_hat'*Y_re   ) ./ ( (  h_hat - sum( b_hat.*alpha_hat )' ) *ones(1,L) );
S(S<0) = 0;

% Obtain Y
Y_Ans = A*S;

% Performance Comparison
time = etime(clock,t0);
percentage = Missing_Percentage(Y)
Y1 = reshape(Y_Ans',100,100,224);
error = Error_Calculation(Y1,X3D)

% plot
%  x = a(1,:); y = a(2,:);
%  scatter(x,y,'filled');

figure;
map1_est= reshape(S(1,:),100,100);
subplot(2,3,1);
imshow(map1_est);title('map 1 est');

map2_est= reshape(S(2,:),100,100);
subplot(2,3,2);
imshow(map2_est);title('map 2 est');
 
map3_est= reshape(S(3,:),100,100);
subplot(2,3,3);
imshow(map3_est);title('map 3 est');
 
subplot(2,3,4);
plot(A(:,1)); title('est signature 1');
axis([1 224 0 1]);

subplot(2,3,5);
plot(A(:,2)); title('est signature 2');
axis([1 224 0 1]);

subplot(2,3,6);
plot(A(:,3)); title('est signature 3');
axis([1 224 0 1]);
