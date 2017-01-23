% May 28 2009
% written by T. K. Pong and P. Tseng
%
% This Matlab function solves
%   min_{W} f(W) = 1/2*\|A*W-B\|^2_F+ mu*\|W\|_*,
% for multiple values of mu>0,
% where A is p by n and B is p by m (so W is n by m), and
% \|\|_* is the nuclear norm.
% The code requires that p>=m.
% 
% This code first reduces A to have full row rank.  Then it
% applies an accelerated proximal gradient method
% to solve the primal problem when mu is large, or
% the dual problem when mu is small.
%
% This matlab function can be called using the command
%
% [Wlist,objlist] = mat_primal_dual(A,B,tol,freq,mulist);
%
% Input:
% A,B are as described above. 
% tol    = termination tolerance (e.g., 1e-3).
% freq   = frequency of termination checks (e.g., n or 10).
% mulist = a column vector of mu values (positive).
%
% Output: 
% Wlist   = a matrix storing in its ith row a solution W for the ith mu value in mulist.
% objlist = a vector storing the objective value of the solutions in Wlist.

function [Wlist,objlist] = mat_primal_dual(A,B,tol,freq,mulist)

n = size(A,2);
m = size(B,2);
p = size(B,1);
fprintf(' m,n,p = %g,%g,%g\n', m,n,p)
nmu = size(mulist,1);
maxiter = 5000;
meps = 1e-8;

Wlist = [];
objlist = [];

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
  fprintf(' reduce A to have full row rank: \n');
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
options.disp = 0;
mu0 = svds(AB,1,'L',options);

t_CE = toc;
fprintf(' done computing C and E, time: %g \n',t_CE);

% Initialization

U = zeros(r,m);
W = zeros(r,m);	
options.disp = 0;
eigmaxM = eigs(M,1,'LM',options);	%compute largest eigenvalue of M
eigmaxC = eigs(C,1,'LM',options);	%compute largest eigenvalue of C
nm = 0;
t_alg = 0;

while (nm < nmu)
  nm = nm+1;
  mu = mulist(nm);
  if mu >= mu0*eigmaxM/(r*eigmaxC/2+eigmaxM) % use primal when mu is large.
    
    iter = 0;
    tic
    
    W0 = W;
    theta = 1;
    theta0 = 1; 
    
    options.disp = 0;
    L = eigmaxM; 	%Lipschitz constant (alg diverges if it's halved)
    % L = norm(M);		%slower than eigs(M,1)
    fprintf(' nm = %g  mu = %g  L = %g  tol = %g  freq = %g\n',nm,mu,L,tol,freq); 
    
    while iter <= maxiter
      
      Y = W + (theta*(1/theta0-1))*(W-W0);
      G = M*Y - AB;
      T = Y - G/L;
      [R,D,S] = svd(T,'econ'); % T=R*D*S'; This and the following line compute
                           % the minimizer to step 2 of accel. grad.
                           % algorithm.
      W0 = W;
      W = R*(max(D-eye(size(D,1))/(L/mu),0))*S';
      
      if norm(W-W0,'fro') <= meps
        fprintf(' termination due to negligible change in U = %g \n',norm(W-W0,'fro'));
        U = M*(W-E);
        [R,D,S] = svd(U,'econ');
        U = R*min(D,mu)*S'; 		%Project to make U dual feasible
        pobj = (mytrace(W,M*W)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
        dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
        fprintf(' iter = %g  dobj = %g  pobj = %g \n',iter,dobj,pobj);
        break
      end
      
      theta0 = theta;
      theta = (sqrt(theta^4+4*theta^2)-theta^2)/2; % Update theta
      iter = iter + 1;
      
      % Compute the gradient of the smooth part
      if (iter > 0) && (mod(iter,freq) == 0)
        U = M*(W-E);
        [R,D,S] = svd(U,'econ');
        U = R*min(D,mu)*S';		%Project to make U dual feasible
        pobj = (mytrace(W,M*W)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
        dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
        fprintf(' iter = %g  dobj = %g  pobj = %g \n',iter,dobj,pobj);
        if abs(pobj-dobj) < tol*(abs(dobj)+1)
          break
        end
      end
    end
    t_alg = t_alg + toc;
    fprintf(' PAPG: mu/iter/cpu/gap = %g/%g/%g/%g\n', mu,iter,t_alg + t_CE + t_redA + t_rankA,abs(pobj-dobj)/(abs(dobj)+1))
  else

    k = 0;
    stop = 0;

    tic
 
    if nm > 1 && mu < mulist(nm-1)
        svmax = svds(U,1);
        if svmax > mu
          U = U*(mu/svmax);	% initialize with U corresp. to W=0, scaled so its sv <= mu.
        end
    end
    
    U0 = U;
    theta = 1;
    theta0 = 1;
    %  L = norm(C);
    options.disp = 0;   % suppress display in eigs
    L = eigmaxC;			% works fine in practice
    fprintf(' nm = %g  mu = %g  L = %g\n', nm,mu,L);
    Wavg = zeros(r,m);		% initialize Wavg to zero.    
    
    while stop == 0 && k <= maxiter
      
      Y = U + (theta*(1/theta0-1))*(U-U0);
      G = C*Y + E; 
      [R,D,S] = svd(Y-G/L,'econ');    % For m<=n, R and D are m by m, S is n by m; otherwise, S and D are n by n, R is m by n.
      U0 = U;
      U = R*min(D,mu)*S';
      Wavg = (1-theta)*Wavg + theta*G;

      % early termination if U-U0 is tiny
      if norm(U-U0,'fro') < meps
        fprintf(' termination due to negligible change in U = %g\n',norm(U-U0,'fro'));
        W = E + C*U;
        pobj = (mytrace(W,M*W)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
        pobjavg = (mytrace(Wavg,M*Wavg)+m3)/2 - mytrace(AB,Wavg) + mu*sum(svd(Wavg));
        dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
        dfeas = max(0,svds(U,1)-mu);
        fprintf(' iter = %g  dobj = %g  dual feas = %g  pobj = %g  pobjavg = %g \n',k,dobj,dfeas,pobj,pobjavg);
        break
      end

      theta0 = theta;
      theta = 2*theta/(sqrt(theta^2+4)+theta);
      k = k + 1;    
      if (k > 0) && (mod(k,freq) == 0)
        W = E + C*U;
        pobj = (mytrace(W,AB+U)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
        pobjavg = (mytrace(Wavg,M*Wavg)+m3)/2 - mytrace(AB,Wavg) + mu*sum(svd(Wavg));
        dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
        dfeas = max(0,svds(U,1)-mu);
        fprintf(' iter = %g  dobj = %g  dual feas = %g  pobj = %g  pobjavg = %g \n',k,dobj,dfeas,pobj,pobjavg);
        if abs(min(pobj,pobjavg)-dobj) < tol*(abs(dobj)+1) && dfeas < tol
          stop = 1;
        end
      end
    end
    t_alg = t_alg + toc;
    fprintf(' DAGP: mu/iter/cpu/gap = %g/%g/%g/%g\n', mu, k,t_alg + t_CE + t_redA + t_rankA, abs(min(pobj,pobjavg)-dobj)/(abs(dobj)+1)')
    if pobjavg < pobj
      W = Wavg;
      pobj = pobjavg;
    end
  end
  if reduce_flag == 1
    W1 = R0*[W;zeros(n-r,m)];
  else
    W1 = W;
  end
  Wlist = [Wlist;vec(W1)'];
  objlist = [objlist;pobj];
end