% May 25 2009
% written by T. K. Pong and P. Tseng
%
% This Matlab function solves
%   min_{W} f(W) = 1/2*\|A*W-B\|^2_F+mu\|W\|_*,
% where A is p by n and B is p by m (so W is n by m), mu>0, and
% \|\|_* is the nuclear norm.
% The code requires that p>=m.
% 
% This code first reduces A to have full row rank.  Then it
% applies an accelerated proximal gradient method
% to solve this problem.
%
% This matlab function can be called using the command
%
% [W,fmin] = mat_primal(A,B,mu,alg,tol,freq,init);
%
% Input:
% A,B,mu are as described above. 
% alg    = 1  Accelerated proximal grad method I with constant stepsize 1/L
%          2  Accelerated proximal grad method II with constant stepsize 1/L
%          3  Accelerated proximal grad method III with constant stepsize 1/L
% tol    = termination tolerance (e.g., 1e-3).
% freq   = frequency of termination checks (e.g., n or 10 or 500).
% init   = 1 to initialize with least-square solution (use when mu is small)  
%          2 to initialize with W=0 (use when mu is large) 
%
% Output: 
% W      = the optimal solution
% pobj   = optimal objective function value of the primal problem.

function [W,pobj] = mat_primal(A,B,mu,alg,tol,freq,init)

n = size(A,2);
m = size(B,2);
p = size(A,1);
if size(B,1) ~= p
  error('A and B must have same number of rows');
end

% termination tolerance & frequency of termination checks
%tol = input(' enter the tolerance (1e-3) ');
fprintf(' n = %g  m = %g  p= %g  tol= %g \n',n,m,p,tol);
maxiter = 5000;		% maximum number of iterations
meps = 1e-8;			% machine epsilon: used for early termination check

% check if A has full row rank
reduce_flag = 1;
if p >= n 
  tic
  r = rank(A);		% rank is expensive, do this only if needed
  t_rankA = toc;
  if r == n 
    reduce_flag = 0;
  end
else t_rankA = 0;
end

% If A lacks full row rank, reduce A to "upper triangular".  
if reduce_flag == 1
%  fprintf(' reduce A to have full row rank: \n');
  tic
  [R0,S0,E0] = qr(A');
  r = rank(S0);
  Anew = (S0(1:r,:))';
  Bnew = E0'*B;
  clear S0 E0
  t_redA = toc;
  fprintf(' done reducing A, time: %g\n', t_redA);
else
  r = n;
  t_redA = 0;
end
 
% From now on, the n in the code and in the description corresponds to the
% rank of Anew, i.e., r.
tic

if reduce_flag == 1 
  M = Anew'*Anew;
  AB = Anew'*Bnew;
  m3 = mytrace(Bnew,Bnew);
else
  M = A'*A;
  AB = A'*B;
  m3 = mytrace(B,B);
end
C = inv(M);
E = C*AB;
f = mytrace(AB*AB',C) - m3;

t_CE = toc;
fprintf(' done computing C and E, time: %g \n',t_CE);

%%%% Accelerated proximal gradient method I %%%%
if alg == 1
  k = 0;
  tic

% Initialize W
  if init == 1
    W = E;		%Least-square initialization.
  else 
    W = zeros(r,m);
  end

  W0 = W;
  theta = 1;
  theta0 = 1;
  options.disp = 0;
  L = eigs(M,1,'LM',options); 	%Lipschitz constant (alg diverges if it's halved)
  %L = norm(M);		%slower than eigs(M,1)
  fprintf(' L = eig(M) = %g , tol = %g, freq = %g \n', L, tol, freq)

  while k <= maxiter
    Y = W + (theta/theta0-theta)*(W-W0);
    G = M*Y - AB;
    [R,D,S] = svd(Y-G/L,'econ');                        
    W0 = W;
    W = R*(max(D-(mu/L)*eye(size(D,1)),0))*S';

    if norm(W-W0,'fro') <= meps
      fprintf(' termination due to negligible change in U = %g \n', norm(W-W0,'fro'));
      U = M*(W-E);
      [R,D,S] = svd(U,'econ');
      U = R*min(D,mu)*S'; 		%Project to make U dual feasible
      pobj = (mytrace(W,M*W)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
      dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
      fprintf(' iter = %g  dobj = %g  pobj = %g  time = %g \n',k,dobj,pobj,toc);
      break
    end
  
    theta0 = theta;
    theta = (sqrt(theta^4+4*theta^2)-theta^2)/2; % Update theta
    k = k + 1;
    if (k > 0) && (mod(k,freq) == 0)
      U = M*(W-E);
      [R,D,S] = svd(U,'econ');
      U = R*min(D,mu)*S';		%Project to make U dual feasible
      pobj = (mytrace(W,M*W)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
      dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
%      fprintf(' iter = %g  dobj = %g  pobj = %g  time = %g \n',k,dobj,pobj,toc);
      fprintf('.');
      if abs(pobj-dobj) < tol*(abs(dobj)+1)
      fprintf('\n iter = %g  dobj = %g  pobj = %g  time = %g \n',k,dobj,pobj,toc);
        break
      end
    end
  end
  t_alg = toc;
  fprintf(' APG I: iter/cpu/gap = %g/%g/%g\n',k,t_rankA+t_redA+t_CE+t_alg,abs(pobj-dobj)/(abs(dobj)+1))
end 

%%%% Accelerated proximal gradient method II %%%%
if alg == 2
  k = 0;
  tic

% Initialize W
  if init == 1
    W = E;		%Least-square initialization.
  else 
    W = zeros(r,m);
  end

  Z = W;
  theta = 1;
  options.disp = 0;
  L = eigs(M,1,'LM',options); 	%Lipschitz constant (alg diverges if it's halved)
  %L = norm(M);		%slower than eigs(M,1)
  fprintf(' L = eig(M) = %g , tol = %g, freq = %g \n', L, tol, freq)

  while k <= maxiter
    Y = (1-theta)*W + theta*Z;
    G = M*Y - AB;
    [R,D,S] = svd(Z-G/(theta*L),'econ');                        
    Z = R*(max(D-eye(size(D,1))/(theta*L/mu),0))*S';
    W0 = W;
    W = (1-theta)*W + theta*Z;

    if norm(W-W0,'fro') <= meps
      fprintf(' termination due to negligible change in U = %g \n', norm(W-W0,'fro'));
      U = M*(W-E);
      [R,D,S] = svd(U,'econ');
      U = R*min(D,mu)*S'; 		%Project to make U dual feasible
      pobj = (mytrace(W,M*W)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
      dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
      fprintf(' iter = %g  dobj = %g  pobj = %g  time = %g \n',k,dobj,pobj,toc);
      break
    end
  
    theta = (sqrt(theta^4+4*theta^2)-theta^2)/2; % Update theta
    k = k + 1;
    if (k > 0) && (mod(k,freq) == 0)
      U = M*(W-E);
      [R,D,S] = svd(U,'econ');
      U = R*min(D,mu)*S';		%Project to make U dual feasible
      pobj = (mytrace(W,M*W)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
      dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
%      fprintf(' iter= %g  dobj= %g  pobj= %g  time = %g \n',k,dobj,pobj,toc);
      fprintf('.');
      if abs(pobj-dobj) < tol*(abs(dobj)+1)
      fprintf('\n iter = %g  dobj = %g  pobj = %g  time = %g \n',k,dobj,pobj,toc);
        break
      end
    end
  end
  t_alg = toc;
  fprintf(' APG II: iter/cpu/gap = %g/%g/%g\n',k,t_rankA + t_redA + t_CE + t_alg,abs(pobj-dobj)/(abs(dobj)+1))
end 


%%%% Accelerated proximal gradient method III %%%%
if alg == 3
  k = 0;
  tic

  Z = zeros(r,m);
  W = Z;
  theta = 1;
  Gsum = zeros(r,m);
  thetasum = 0;

  options.disp = 0;
  L = eigs(M,1,'LM',options); 	%Lipschitz constant (alg diverges if it's halved)
  %L = norm(M);		%slower than eigs(M,1)
  fprintf(' L = %g , tol = %g, freq = %g \n', L, tol, freq)

  while k <= maxiter

    Y = (1-theta)*W + theta*Z;
    G = M*Y-AB;
    Gsum = Gsum + G/theta;
    thetasum = thetasum + 1/theta;
    [R,D,S] = svd(-Gsum/L,'econ');                        
    W0 = W;
    Z = R*(max(D-(mu*thetasum/L)*eye(size(D,1)),0))*S';
    W = (1-theta)*W + theta*Z;

    if norm(W-W0,'fro') <= meps
      fprintf(' termination due to negligible change in U = %g \n',norm(W-W0,'fro'));
      U = M*(W-E);
      [R,D,S] = svd(U,'econ');
      U = R*min(D,mu)*S'; 		%Project to make U dual feasible
      pobj = (mytrace(W,M*W)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
      dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
      fprintf(' iter= %g  dobj= %g  pobj= %g  time = %g \n',k,dobj,pobj,toc);
      break
    end
  
    theta = (sqrt(theta^4+4*theta^2)-theta^2)/2; % Update theta
    k = k + 1;
    if (k > 0) && (mod(k,freq) == 0)
      U = M*(W-E);
      [R,D,S] = svd(U,'econ');
      U = R*min(D,mu)*S';		%Project to make U dual feasible
      pobj = (mytrace(W,M*W)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
      dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
 %     fprintf(' iter = %g  dobj = %g  pobj = %g  time = %g \n',k,dobj,pobj,toc);
      fprintf('.');
      if abs(pobj-dobj) < tol*(abs(dobj)+1)
        fprintf('\n iter = %g  dobj = %g  pobj = %g  time = %g \n',k,dobj,pobj,toc);
        break
      end
    end
  end
  t_alg = toc;
  fprintf(' APG III: iter/cpu/gap = %g/%g/%g\n',k,t_rankA + t_redA + t_CE + t_alg,abs(pobj-dobj)/(abs(dobj)+1))
end 


if reduce_flag == 1
  W = R0*[W;zeros(n-r,m)];
end