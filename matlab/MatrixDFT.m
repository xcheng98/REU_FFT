function [FFTX] = MatrixDFT(X)
x = reshape(X,1,[]);
N = length(x);
y = zeros(N,1);
W = zeros(N,N);

for k=1:N
    for n=1:N
    W(n,k) = exp(-2*pi*1i*(k-1)*(n-1)/N);
    end
end

y = x*W;
FFTX = y;