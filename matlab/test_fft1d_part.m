A = 64;
B = 64;
N = A*B;
nvec = A*B;
X = rand(A*B*nvec,1) + sqrt(-1)*rand(A*B*nvec,1);
X = reshape( X, [A*B,nvec]);

tic;
Xhat = fft1d_part( A,N,nvec, X );
toc

tic;
fftX = fft( reshape(X, [A*B,nvec]));
toc
fftX = fftX(1:A,1:nvec);

diff = norm( fftX-Xhat,1);
disp(sprintf('difference in fft1d_part: diff=%e', diff ));
