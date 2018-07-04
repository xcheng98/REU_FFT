function Xhat = fft2d_part(A,N, X )
%
% Xhat = fft2d_part(A,N, X )
% 
% compute only the top A by A  number of Fourier values from N by N data
%

isok = (mod(numel(X),N) == 0);
if (~isok),
  error(sprintf('fft2d_part: numel(X)=%d, N=%d', ...
                             numel(X),    N ));
  return;
end;

% ---------
% 1st stage
% ---------
nvec = round(numel(X)/N);
Xhat1 = fft1d_part( A, N, nvec, reshape(X, [N,nvec]));

nvec2 = A;
Xhat = fft1d_part( A, N, nvec2, transpose( Xhat1 ) );
Xhat = transpose(Xhat);
