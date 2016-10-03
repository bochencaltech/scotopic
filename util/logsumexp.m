function A = logsumexp(A, dim, b)
% A = logsumexp(A, dim, b)
% compute the numerically-safe version of logsumexp along dimension dim.
% [.dim] dimension along which to compute the sum [1]
% [.b] the base for log and exp [e]
%
% Equalities:
%  b^x = (e^x)^ (log(b)) 
%  log_b(x) = log(x)/log(b)
%  res = A*+log_b(sum(b^(A-A*))) = A*+log(sum(b^(A-A*)))/log(b)
if ~exist('dim','var'), dim = 1; end
if ~exist('b','var'), b = exp(1); end
maxA = max(A, [], dim);
A = maxA + log( sum(b.^bsxfun(@minus, A, maxA), dim) )/log(b);
end