function[y] = SimpleDFT(X)
x = reshape(X,1,[]);
N = length(x);
y = zeros(N,1);

for k=1:N
    for n=1:N
        y(k) = y(k)+x(n)*exp(-2*pi*i*(k-1)*(n-1)/N);
    end
end