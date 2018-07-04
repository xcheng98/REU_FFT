% test fft2d_part

A = 64;
B = 64;
N = A*B;
X = rand(N,N) + sqrt(-1)*rand(N,N);

tic;
Xhat = fft2d_part(A,N, X );
toc;

tic
fftX = fft2( X, N,N);
toc

diff = norm( fftX(1:A,1:A)-Xhat(1:A,1:A),1);
disp(sprintf('difference of fft2d_part: diff=%e', diff ));
