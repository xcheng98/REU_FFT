% Simple file to test various fft functions
n = 1024;
X_re = 2*rand(1, n)-1;
X_im = 2*rand(1, n)-1;
X = X_re + sqrt(-1)*X_im;

%In-built Function
tic;
FX = fft(X);
timeinbuilt = toc;

%Simple DFT Code using equation
tic;
F2X = SimpleDFT(X);
timesimple = toc;

%DFT Code by constructing matrix
tic;
F3X = MatrixDFT(X);
timematrix = toc;

%DFT Code by constructing matrix using Lookup Table
tic;
F4X = MatrixDFTLookUp(X);
timematrixlookup = toc;

%DFT Code where real and imaginnary i/p given seperately
tic;
[FX5_r, FX5_i] = MatrixDFT_2ip(X_re,X_im);
timematrix2ip = toc;

%DFT Code where i/p given seperately and 3M combination is used
tic;
[FX6_r, FX6_i] = MatrixDFT_2ip_3M(X_re,X_im);
timemy3M = toc;

%FFT with recursive buttery equation used
tic;
F7X = Radix2fft(X);
timeradix2 = toc;

%FFT code given by Dr. Ed
tic;
[FX_re, FX_im] = fft2( X_re, X_im, 1);
timefft2= toc;

%To ensure all ffts are correct (FX23 can be replaced with the various
%ffts)

FX6 = FX6_r + sqrt(-1)*FX6_i;
diff = abs(FX - FX6);
err=sum(diff);
  
  