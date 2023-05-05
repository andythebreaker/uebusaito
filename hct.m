function [A_est, time, a] = HyperCSI(X,N)
t0 = clock;
%%PCA Dimension Reduction
[M L ] = size(X);
row_mean_pca = mean(X,2);  %mean
U(1:224,:) = X(1:224,:)-row_mean_pca;

q1_idx=0;
q2_idx=0;
[V,D] = eig(U*U');
for i=1:224
   if D(i,i) > q1_idx
       q1_idx =i;
   end
end
D(q1_idx,q1_idx)=0;
for i=1:224
   if D(i,i) > q2_idx
       q2_idx = i;
   end
end
C(:,1)=V(:,q1_idx);
C(:,2)=V(:,q2_idx);
a = C'*U;
%=======================================
%%SPA Find the vertex
%Find frist vertex
tempnorm=0;
a_idx=0;
for i=1:L
    if norm(a(:,i)) > tempnorm
        tempnorm = norm(a(:,i));
        a_idx = i;
    end
end
a_spa(:,1) = a(:,a_idx);

%Find Second vertex
%generate projection matrix
PM = eye(2)-a_spa(:,1)*a_spa(:,1)' / (norm(a_spa(:,1))*norm(a_spa(:,1)));

tempnorm=0;
for j=1:L
    a_p(:,j) = PM*a(:,j);
    if norm(a_p(:,j)) > tempnorm
        tempnorm = norm(a_p(:,j));
        a_idx = j;
    end
end
a_spa(:,2) = a(:,a_idx);
a2_idx=a_idx;

%Find Third vertex
tempnorm=0;
for k=1:L
    a_p(:,k) = a_p(:,k)-a_p(:,a2_idx);
    if norm(a_p(:,k)) > tempnorm
        tempnorm = norm(a_p(:,k));
        a_idx = k;
    end
end
a_spa(:,3) = a(:,a_idx);
%=============================================
%Normal Vector Estimation
vd1 = a_spa(:,2)-a_spa(:,3);
vd2 = a_spa(:,3)-a_spa(:,1);
vd3 = a_spa(:,1)-a_spa(:,2);
b1_tilde = nv(vd1);
b2_tilde = nv(vd2);
b3_tilde = nv(vd3);
r1=norm(a_spa(:,2)-a_spa(:,3))*0.3;
r2=norm(a_spa(:,3)-a_spa(:,1))*0.3;
r3=norm(a_spa(:,1)-a_spa(:,2))*0.3;
max=0;max2=0;
%h1_hat of Normal Vector
for i=1:L
    v2 = a(:,i)-a_spa(:,2);
    v3 = a(:,i)-a_spa(:,3);
    if  norm(v2)<r2                        %testing point in given circle range
        if dot(a(:,i),b1_tilde)>max
            max = dot(a(:,i),b1_tilde);
            p1_idx1=i;
        end
    end
    if norm(v3)<r3                          %testing point in given circle range
        if dot(a(:,i),b1_tilde)>max2
            max2 = dot(a(:,i),b1_tilde);
            p2_idx1=i;
        end
    end
end
p1=a(:,p1_idx1)-a(:,p2_idx1);
b1_hat=nv(p1);

h1_hat=0;
for i=1:L
    m=b1_hat'*a(:,i);
    if m>h1_hat
        h1_hat = m;
    end
end
h1_hat=h1_hat/norm(b1_hat);
%h2_hat of Normal Vector
max=0;max2=0;
for i=1:L
    v3 = a(:,i)-a_spa(:,3);
    v1 = a(:,i)-a_spa(:,1);
    if norm(v3)<r3
        if dot(a(:,i),b2_tilde)>max
            max = dot(a(:,i),b2_tilde);
            p1_idx2=i;
        end
    end
    if norm(v1)<r1
        if dot(a(:,i),b2_tilde)>max2
            max2 = dot(a(:,i),b2_tilde);
            p2_idx2=i;
        end
    end
end
p2=a(:,p1_idx2)-a(:,p2_idx2);
b2_hat=nv(p2);
h2_hat=0;
for i=1:L
    m=b2_hat'*a(:,i);
    if m>h2_hat
        h2_hat = m;
    end
end
h2_hat=h2_hat/norm(b2_hat);
%h3_hat of Normal Vector
max=0;max2=0;
for i=1:L
    v1 = a(:,i)-a_spa(:,1);
    v2 = a(:,i)-a_spa(:,2);
    if  norm(v1)<r1
        if dot(a(:,i),b3_tilde)>max
            max = dot(a(:,i),b3_tilde);
            p1_idx3=i;
        end
    end
    if norm(v2)<r2
        if dot(a(:,i),b3_tilde)>max2
            max2 = dot(a(:,i),b3_tilde);
            p2_idx3=i;
        end
    end
end
p3=a(:,p1_idx3)-a(:,p2_idx3);
b3_hat=nv(p3);

h3_hat=0;
for i=1:L
    m=b3_hat'*a(:,i);
    if m>h3_hat
        h3_hat = m;
    end
end
h3_hat=h3_hat/norm(b3_hat);
%Find the Exact Vertex by Solving Simultaneous Equations
c1=h1_hat*norm(b1_hat);
c2=h2_hat*norm(b2_hat);
c3=h3_hat*norm(b3_hat);
A1(1,:)=b2_hat';
A1(2,:)=b3_hat';
A2(1,:)=b3_hat';
A2(2,:)=b1_hat';
A3(1,:)=b1_hat';
A3(2,:)=b2_hat';
B1=[c2,c3]';
B2=[c3,c1]';
B3=[c1,c2]';
P1=inv(A1)*B1;
P2=inv(A2)*B2;
P3=inv(A3)*B3;
%PCA inverse operation
A_est(:,1)=C*P1+row_mean_pca;
A_est(:,2)=C*P2+row_mean_pca;
A_est(:,3)=C*P3+row_mean_pca;
time = etime(clock,t0);
end
