function [A_est, S_est, time] = HyperCSI(X,N)
t0 = clock;
%PCA Dimension Reduction
row_mean_pca = mean(X,2);  %mean
for i=1:10000
    U(1:224,i) = X(1:224,i)-row_mean_pca(1:224);
end
q1_idx=0;
q2_idx=0;
[V,D] = eig(U*transpose(U));
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
C(1:224,1)=V(1:224,q1_idx);
C(1:224,2)=V(1:224,q2_idx);
a = transpose(C)*U;
%=======================================
%SPA Find the vertex
%Find frist vertex
tempnorm=0;
a_idx=0;
for i=1:10000
    if norm(a(1:2,i)) > tempnorm
        tempnorm = norm(a(1:2,i));
        a_idx = i;
    end
end
a_spa(1:2,1) = a(1:2,a_idx);

%Find Second vertex
PM = eye(2)-a_spa(1:2,1)*transpose(a_spa(1:2,1))/(norm(a_spa(1:2,1))*norm(a_spa(1:2,1)));
%generate projection matrix
tempnorm=0;
for j=1:10000
    a_p(1:2,j) = PM*a(1:2,j);
    if norm(a_p(1:2,j)) > tempnorm
        tempnorm = norm(a_p(1:2,j));
        a_idx = j;
    end
end
a_spa(1:2,2) = a(1:2,a_idx);
a2_idx=a_idx;

%Find Third vertex
tempnorm=0;
for k=1:10000
    a_p(1:2,k) = a_p(1:2,k)-a_p(1:2,a2_idx);
    if norm(a_p(1:2,k)) > tempnorm
        tempnorm = norm(a_p(1:2,k));
        a_idx = k;
    end
end
a_spa(1:2,3) = a(1:2,a_idx);
%=============================================
%Normal Vector Estimation
vd1 = a_spa(1:2,2)-a_spa(1:2,3);
vd2 = a_spa(1:2,3)-a_spa(1:2,1);
vd3 = a_spa(1:2,1)-a_spa(1:2,2);
b1_tilde(1,1)=-vd1(2,1);
b1_tilde(2,1)=vd1(1,1);
b2_tilde(1,1)=-vd2(2,1);
b2_tilde(2,1)=vd2(1,1);
b3_tilde(1,1)=-vd3(2,1);
b3_tilde(2,1)=vd3(1,1);
r1=norm(a_spa(1:2,2)-a_spa(1:2,3))*0.3;
r2=norm(a_spa(1:2,3)-a_spa(1:2,1))*0.3;
r3=norm(a_spa(1:2,1)-a_spa(1:2,2))*0.3;
max=0;max2=0;
%h1_hat of Normal Vector
for i=1:10000
    v2 = a(1:2,i)-a_spa(1:2,2);
    v3 = a(1:2,i)-a_spa(1:2,3);
    if  norm(v2)<r2                        %testing point in given circle range
        if dot(a(1:2,i),b1_tilde)>max
            max = dot(a(1:2,i),b1_tilde);
            p1_idx1=i;
        end
    end
    if norm(v3)<r3                          %testing point in given circle range
        if dot(a(1:2,i),b1_tilde)>max2
            max2 = dot(a(1:2,i),b1_tilde);
            p2_idx1=i;
        end
    end
end
p1=a(1:2,p1_idx1)-a(1:2,p2_idx1);
b1_hat(1,1)=-p1(2,1);
b1_hat(2,1)=p1(1,1);
h1_hat=0;
for i=1:10000
    m=transpose(b1_hat)*a(1:2,i);
    if m>h1_hat
        h1_hat = m;
    end
end
h1_hat=h1_hat/norm(b1_hat);
%h2_hat of Normal Vector
max=0;max2=0;
for i=1:10000
    v3 = a(1:2,i)-a_spa(1:2,3);
    v1 = a(1:2,i)-a_spa(1:2,1);
    if norm(v3)<r3
        if dot(a(1:2,i),b2_tilde)>max
            max = dot(a(1:2,i),b2_tilde);
            p1_idx2=i;
        end
    end
    if norm(v1)<r1
        if dot(a(1:2,i),b2_tilde)>max2
            max2 = dot(a(1:2,i),b2_tilde);
            p2_idx2=i;
        end
    end
end
p2=a(1:2,p1_idx2)-a(1:2,p2_idx2);
b2_hat(1,1)=-p2(2,1);
b2_hat(2,1)=p2(1,1);
h2_hat=0;
for i=1:10000
    m=transpose(b2_hat)*a(1:2,i);
    if m>h2_hat
        h2_hat = m;
    end
end
h2_hat=h2_hat/norm(b2_hat);
%h3_hat of Normal Vector
max=0;max2=0;
for i=1:10000
    v1 = a(1:2,i)-a_spa(1:2,1);
    v2 = a(1:2,i)-a_spa(1:2,2);
    if  norm(v1)<r1
        if dot(a(1:2,i),b3_tilde)>max
            max = dot(a(1:2,i),b3_tilde);
            p1_idx3=i;
        end
    end
    if norm(v2)<r2
        if dot(a(1:2,i),b3_tilde)>max2
            max2 = dot(a(1:2,i),b3_tilde);
            p2_idx3=i;
        end
    end
end
p3=a(1:2,p1_idx3)-a(1:2,p2_idx3);
b3_hat(1,1)=-p3(2,1);
b3_hat(2,1)=p3(1,1);
h3_hat=0;
for i=1:10000
    m=transpose(b3_hat)*a(1:2,i);
    if m>h3_hat
        h3_hat = m;
    end
end
h3_hat=h3_hat/norm(b3_hat);
%Find the Exact Vertex by Solving Simultaneous Equations
c1=h1_hat*norm(b1_hat);
c2=h2_hat*norm(b2_hat);
c3=h3_hat*norm(b3_hat);
A1(1,1:2)=transpose(b2_hat);
A1(2,1:2)=transpose(b3_hat);
A2(1,1:2)=transpose(b3_hat);
A2(2,1:2)=transpose(b1_hat);
A3(1,1:2)=transpose(b1_hat);
A3(2,1:2)=transpose(b2_hat);
B1=transpose([c2,c3]);
B2=transpose([c3,c1]);
B3=transpose([c1,c2]);
P1=inv(A1)*B1;
P2=inv(A2)*B2;
P3=inv(A3)*B3;
%PCA inverse operation
A_est(1:224,1)=C*P1+row_mean_pca;
A_est(1:224,2)=C*P2+row_mean_pca;
A_est(1:224,3)=C*P3+row_mean_pca;
%Abundance maps
d_a1=abs((transpose(b1_hat)*P1-c1)/norm(b1_hat));
d_a2=abs((transpose(b2_hat)*P2-c2)/norm(b2_hat));
d_a3=abs((transpose(b3_hat)*P3-c3)/norm(b3_hat));
for i=1:10000
    d1_p=abs((transpose(b1_hat)*a(1:2,i)-c1)/norm(b1_hat));
    S_est(1,i)=d1_p/d_a1;
    d2_p=abs((transpose(b2_hat)*a(1:2,i)-c2)/norm(b2_hat));
    S_est(2,i)=d2_p/d_a2;
    d3_p=abs((transpose(b3_hat)*a(1:2,i)-c3)/norm(b3_hat));
    S_est(3,i)=d3_p/d_a3;
end
time = etime(clock,t0);
