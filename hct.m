t0 = clock;
% Algorithm
%load Nevada
X3D=X;
whz=size(X3D);
M=whz(3);
L=whz(1)*whz(2);
X = reshape(X3D,L,M)'; % 3D to 2D

%Generate missing data(Revise Here)
Y = X;
pixelcut = [1:600 ];%, 1767:2500 , 2710:2722 , 2800:3500 , 3520 , 3545:3596 , 3600 , 3710 , 3800:10000]; 
bandcut = [1:2 ];%, 223:224];                                                                           
Y(bandcut , pixelcut) =0;
%showrgbhyper_2d(Y,N);

% Subset 
Y_Omega = X;
Y_Omega(: , pixelcut) = [];
 
% Obtain A by HyperCSI
[~ , L] = size(Y_Omega);        
[A, timeHyperCSI,a] = hpycsi(Y_Omega,N,false);

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
percentage = Missing_Percentage(Y);
Y1 = reshape(Y_Ans',whz(1),whz(2),M);
error = frobenius_norm(Y1,X3D);

pa=percentage;
er=error;

%% plt
if(pltendbool==true)
figure;
for i = 1:N
    map_est = reshape(S(i,:), whz(1), whz(2));
    subplot(2, N, i);
    imshow(map_est);
    title(['map ' num2str(i) ' est']);
end

for i = 1:N
    subplot(2, N, i+N);
    plot(A(:,i));
    title(['est signature ' num2str(i)]);
    axis([1 M 0 1]);
end
end
