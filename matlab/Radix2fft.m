%Radix 2 FFT based on butterfly diagram

function [y] = Radix2fft(X)

x=reshape(X,1,[]);
N=length(x);
p=nextpow2(length(x));
x= [x zeros(1,(2^p)-length(x))];
S=log2(N);
li=sqrt(-1);
x=bitrevorder(x);
jump=1;

for stage=1:S
    for i = 0:(2^stage):N-1
        for k = 0:(jump-1)
            twiddle= exp(-2*pi*li*(2^(S-stage))*k/N);
            index=k+i+1;
            a=x(index)+x(index+jump).*twiddle;
            b=x(index)-x(index+jump).*twiddle;
            x(index)=a;
            x(index+jump)=b;
        end
    end
    jump=2*jump;
end
y=x;
