function [FX_re, FX_im] = fft2( X_re, X_im, use_combine_in )
% [FX_re, FX_im] = fft2( X_re, X_im  [,use_combine_in] )
%
% perform length 2 FFT 
% input given as real part and imaginary part separately
%
% ----------------------------
% F2_re = real( fft(eye(2,2)) )
% F2_im = imag( fft(eye(2,2)) )
% ----------------------------
idebug = 0;
if (idebug >= 1),
  disp(sprintf('fft2: size(X_re) = (%d,%d)', ...
         size(X_re,1), size(X_re,2) ...
     ));
end;

use_combine = 1;
if (nargin >= 3),
  use_combine = use_combine_in;
end;

F2_re = [ ...
       1,  1; ...
       1, -1; ...
       ];
F2_im = [ ...
       0,  0; ...
       0,  0; ...
       ];

% ---------------------------
% Note very special case that
% F2_re contains only 1 or -1
% F2_im is the zero matrix
% ---------------------------

isok = ...
       (mod(prod(size(X_re)),2) == 0) && ...
       (mod(prod(size(X_im)),2) == 0) && ...
       (prod(size(X_im)) == prod(size(X_re)));
if (~isok),
  error( sprintf('fft2: invalid sizes, size(X_re)=(%d,%d), size(X_im)=(%d,%d)', ...
          size(X_re,1), size(X_re,2), ...
          size(X_im,1), size(X_im,2)   ));
  return;
end;


X_re = reshape( X_re, 2,prod(size(X_re))/2 );
X_im = reshape( X_im, 2,prod(size(X_im))/2 );

% -------------
% F2_re = [1, 1;
%          1, -1]
% -------------
if (use_combine),
  FX_re = F2_re * X_re;
  FX_im = F2_re * X_im;
else
  X_re_1 = X_re(1,:);
  X_re_2 = X_re(2,:);
  X_im_1 = X_im(1,:);
  X_im_2 = X_im(2,:);

  FX_re = [X_re_1 + X_re_2; ...
           X_re_1 - X_re_2];

  FX_im = [X_im_1 + X_im_2; ...
           X_im_1 - X_im_2];
end;
  