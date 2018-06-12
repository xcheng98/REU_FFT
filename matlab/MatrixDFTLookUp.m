function [y] = MatrixDFTLookUp(X)
x = reshape(X,1,[]);
N = length(x);
y = zeros(N,1);
W = zeros(N,N);
%Define C value
c = 128;
N2 = N*N;
%Calculating values of k1 & k2
k1max = ((N2)-mod(N2,c))/c;
w_c_table = zeros(1,k1max+1);
w_table = zeros(1,c);

%Lookup table formulation
WC = exp(-2*pi*i*c/N);

for kone=0:k1max
    w_c_table(kone+1)=(WC)^(kone);
end

for ktwo=0:c-1
    w_table(ktwo+1) = exp(-2*pi*i*(ktwo)/N);
end
    
for l=1:N
    for n=1:N
        k = (l-1)*(n-1);
        k2 = mod(k,c);
        k1 = (k-k2)/c;
        W(n,l) = w_c_table(k1+1)*w_table(k2+1);
    end
end


y = x*W;