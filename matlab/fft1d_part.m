function [Xhat] = fft1d_part( A,N,nvec, X )
% [Xhat] = fft1d_part( A,N,nvec, X )
%
% perform length N FFT  where N  = A*B
% but need only top A coefficients
%
idebug = 0;

B = N/A;
isok = (mod(N,A) == 0);
if (~isok),
  error(sprintf('fft1d_part: N=%d, A=%d, B = %d', ...
                             N,    A,    B ));
  return;
end;

isok = (numel(X) == (A*B*nvec));
if (~isok),
  error(sprintf('fft1d_part: A=%d, B=%d, nvec=%d, numel(X)=%d', ...
                             A,    B,    nvec,    numel(X) ));
  return;
end;

X = reshape(X, [B,A,nvec]);

% ---------------------
% multiple A-length FFT
% ---------------------
X = permute( X, [2,3,1]);  %  shape of X is  now [A, nvec, B]
X = reshape( X, [A, nvec*B]);  

FX = fft(X);  
FX = reshape( FX, [A,nvec,B]); % FX(ahat, ivec, b)

b = 0:(B-1);
zi = sqrt(-1);
eb = reshape(exp( -2*pi*zi*b/(A*B) ), B,1);
ebvec = ones(B,1);

ioff = 1;
Xhat = zeros(A,nvec);

for ahat=0:(A-1),
  % ----------------------
  % matrix vector multiply
  % ----------------------
  xhat_ahat = reshape(FX(ahat+ioff, 1:nvec,:), nvec,B) * ebvec;

  Xhat(ioff+ahat,1:nvec) = reshape( xhat_ahat,1,nvec);
  ebvec = ebvec .* eb;
end;

if (idebug >= 1),
  % ------------
  % check result
  % ------------
  fftX = fft( reshape(X, [A*B,nvec]));
  fftX = fftX(1:A,1:nvec);

  diff = norm( fftX-Xhat,1);
  disp(sprintf('fft1d_part: diff=%e', diff ));
end;
  



