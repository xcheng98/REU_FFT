function [y_r,y_i] = MatrixDFT_2ip_3M(X_r,X_i)
x_r = reshape(X_r,1,[]);
x_i = reshape(X_i,1,[]);
N = length(x_r);
y_r = zeros(N,1);
y_i = zeros(N,1);
W_r = zeros(N,N);
W_r = zeros(N,N);

for k=1:N
    for n=1:N
    W_r(n,k) = cos(-2*pi*(k-1)*(n-1)/N);
    W_i(n,k) = sin(-2*pi*(k-1)*(n-1)/N);
    end
end

T1 = x_r*W_r;
T2 = x_i*W_i;

y_r = T1-T2;
y_i = (x_r+x_i)*(W_r+W_i)-T1-T2;

