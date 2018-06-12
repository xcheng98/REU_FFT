function [y_r,y_i] = MatrixDFT_2ip(X_r,X_i)
x_r = reshape(X_r,1,[]);
x_i = reshape(X_i,1,[]);
N = length(x_r);
y_r = zeros(N,1);
y_i = zeros(N,1);
W_r = zeros(N,N);
W_i = zeros(N,N);

for k=1:N
    for n=1:N
    W_r(n,k) = cos(-2*pi*(k-1)*(n-1)/N);
    W_i(n,k) = sin(-2*pi*(k-1)*(n-1)/N);
    end
end

y_r = x_r*W_r - x_i*W_i;
y_i = x_i*W_r + x_r*W_i;