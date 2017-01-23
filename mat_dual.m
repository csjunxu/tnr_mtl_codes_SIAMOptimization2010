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
% applies a feasible descent method to solve the dual
% problem:
%   min_U 1/2*<C,U*U'> + <E,U>  s.t.   U'*U <= mu^2 I,
% where C=M^{-1}, M=A'*A, and E=C*A'*B (so U is n by m).
% From the solution U we obtain  W=E+C*U (or U=M*(W-E)).
%
% This matlab function can be called using the command
%
% [W,fmin] = mat_dual(A,B,mu,alg,tol,freq,init);
%
% Input:
% A,B,mu are as described above.
% alg    = 1  conditional gradient (Frank-Wolfe) method with line search
%          2  gradient-projection method with large constant stepsize + line search
%          3  gradient-projection method with constant stepsize 0.5/L
%          4  Accelerated grad-proj method I with constant stepsize 1/L
%          5  Accelerated grad-proj method II with constant stepsize 1/L
%          6  Accelerated grad-proj method III with constant stepsize 1/L
% tol    = termination tolerance (e.g., 1e-3).
% freq   = frequency of termination checks (e.g., n or 10 or 500).
% init   = 1 initial U=0, corresp. to W being least square solution (use when mu is small)
%	         2 initial U corresp. to W=0, projected onto U'*U <= mu^2 I (use when mu is large)
%          3 initial U=-mu*A'*B/sigma(A'*B), corresp. to
%            W=0, scaled to satisfy U'*U <= mu^2 I (use when mu is large)
%
% Output:
% W      = the optimal solution
% pobj   = optimal objective function value of the original problem.

function [W,pobj] = mat_dual(A,B,mu,alg,tol,freq,init)

% Read data parameters
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
    nold = n;
    n = r;
    t_redA = toc;
    fprintf(' done reducing A, time: %g\n', t_redA);
else
    t_redA = 0;
end

% From now on, the n in the code and in the description corresponds to the
% rank of Anew, i.e., r.
tic

if reduce_flag == 1
    M = Anew'*Anew;
    AB = Anew'*Bnew;
    m3 = mytrace(Bnew,Bnew);
    C = inv(M);
    E = C*AB;
    f = mytrace(Bnew*E',Anew) - m3;
else
    M = A'*A;
    AB = A'*B;
    m3 = mytrace(B,B);
    C = inv(M);
    E = C*AB;
    f = mytrace(B*E',A) - m3;
end
t_CE = toc;
fprintf(' done computing C and E, time: %g \n',t_CE);

%%%% Conditional gradient method with line search %%%%
if alg == 1
    k = 0;
    stop = 0;
    tic
    
    if init == 1
        U = zeros(n,m);		% initialize with U=0.
    else
        U = -AB;
        if init == 2
            [R,D,S] = svd(U,'econ');
            U = R*min(D,mu)*S';
        else
            svmax = svds(U,1);
            if svmax > mu
                U = U*(mu/svmax);	% initialize with U corresp. to W=0, scaled so its sv <= mu.
            end
        end
    end
    
    options.disp = 0;  		% suppress display in eigs
    while stop == 0 && k <= maxiter
        G = C*U + E;
        [R,D,S] = svd(G,'econ');    % For m<=n, R and D are m by m, S is n by m;
        % otherwise, S and D are n by n, R is m by n.
        hU = -mu*(R*S');
        Del = hU - U;     		% Del is the search direction
        alpha = min(1,-mytrace(G,Del)/mytrace(C*Del,Del));
        
        % early termination if stepsize goes negative
        if alpha < meps
            fprintf(' termination due to negative stepsize = %g \n',alpha);
            W = E + C*U;
            pobj = (mytrace(W,AB+U)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
            dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
            dfeas = max(0,svds(U,1)-mu);
            fprintf(' iter= %g  dobj= %g  dual feas= %g  pobj=  %g  time= %g \n',k,dobj,dfeas,pobj,toc);
            break
        end
        
        U = U + alpha*Del;
        k = k + 1;
        if (k > 0) && (mod(k,freq) == 0)
            W = E + C*U;
            pobj = (mytrace(W,AB+U)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
            dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
            dfeas = max(0,svds(U,1)-mu);
            %         fprintf(' iter= %g  dobj= %g  dual feas= %g  pobj= %g  time= %g \n',k,dobj,dfeas,pobj,toc);
            fprintf('.');
            if abs(pobj-dobj) < tol*(abs(dobj)+1) && dfeas<tol
                fprintf('\n iter = %g  dobj = %g  dual feas = %g  pobj = %g  time = %g \n',k,dobj,dfeas,pobj,toc);
                stop = 1;
            end
        end
    end
    t_alg = toc;
    fprintf(' Frank-Wolfe: iter/cpu/gap = %g/%g/%g\n',k,t_rankA + t_redA + t_CE + t_alg,abs(pobj-dobj)/(abs(dobj)+1))
end

%%%% Gradient-projection method with line search %%%%
if alg == 2
    k = 0;
    stop = 0;
    tic
    
    if init == 1
        U = zeros(n,m);		% initialize with U=0.
    else
        U = -AB;
        if init == 2
            [R,D,S] = svd(U,'econ');
            U = R*min(D,mu)*S';
        else
            svmax = svds(U,1);
            if svmax > mu
                U = U*(mu/svmax);	% initialize with U corresp. to W=0, scaled so its sv <= mu.
            end
        end
    end
    
    
    options.disp = 0;  % suppress display in eigs
    L = eigs(C,1,'LM',options)/8;
    
    while stop == 0 && k <= maxiter
        
        G = C*U + E;
        [R,D,S] = svd(U-G/L,'econ');    % For m<=n, R and D are m by m, S is n by m; otherwise, S and D are n by n, R is m by n.
        hU = R*min(D,mu)*S';
        Del = hU - U;           % Del is the search direction
        alpha = min(1,-mytrace(G,Del)/mytrace(C*Del,Del));
        
        % early termination if stepsize goes negative
        if alpha < meps
            fprintf(' termination due to negative stepsize = %g \n',alpha);
            W = E + C*U;
            pobj = (mytrace(W,AB+U)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
            dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
            dfeas = max(0,svds(U,1)-mu);
            fprintf(' iter= %g  dobj= %g  dual feas= %g  pobj= %g  time= %g \n',k,dobj,dfeas,pobj,toc);
            break
        end
        
        U = U + alpha*Del;
        k = k + 1;
        if (k > 0) && (mod(k,freq) == 0)
            W = E + C*U;
            pobj = (mytrace(W,AB+U)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
            dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
            dfeas = max(0,svds(U,1)-mu);
            %        fprintf(' iter= %g  dobj= %g  dual feas= %g  pobj= %g  time= %g \n',k,dobj,dfeas,pobj,toc);
            fprintf('.');
            if abs(pobj-dobj) < tol*(abs(dobj)+1) && dfeas < tol
                fprintf('\n iter = %g  dobj = %g  dual feas = %g  pobj = %g  time = %g \n',k,dobj,dfeas,pobj,toc);
                stop = 1;
            end
        end
    end
    t_alg = toc;
    fprintf(' GP+LS: iter/cpu/gap = %g/%g/%g\n',k,t_rankA + t_redA + t_CE + t_alg,abs(pobj-dobj)/(abs(dobj)+1))
end

%%%% Gradient-projection method with constant stepsize %%%%
if alg == 3
    k = 0;
    stop = 0;
    tic
    
    if init == 1
        U = zeros(n,m);		% initialize with U=0.
    else
        U = -AB;
        if init == 2
            [R,D,S] = svd(U,'econ');
            U = R*min(D,mu)*S';
        else
            svmax = svds(U,1);
            if svmax > mu
                U = U*(mu/svmax);	% initialize with U corresp. to W=0, scaled so its sv <= mu.
            end
        end
    end
    
    options.disp = 0;   % suppress display in eigs
    % For convergence  L > norm(C)/2;
    L = eigs(C,1,'LM',options)/1.95;
    fprintf(' L = %g\n', L);
    
    while stop == 0 && k <= maxiter
        U0 = U;
        G = C*U + E;
        [R,D,S] = svd(U-G/L,'econ');    % For m<=n, R and D are m by m, S is n by m;
        %otherwise, S and D are n by n, R is m by n.
        U = R*min(D,mu)*S';
        
        % early termination if U-U0 is tiny
        
        if norm(U-U0,'fro')<meps
            fprintf(' termination due to negligible change in U = %g\n',norm(U-U0,'fro'));
            W = E + C*U;
            pobj = (mytrace(W,AB+U)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
            dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
            dfeas = max(0,svds(U,1)-mu);
            fprintf(' iter = %g  dobj = %g  dual feas = %g  pobj = %g  time = %g \n',k,dobj,dfeas,pobj,toc);
            break
        end
        
        k = k + 1;
        if (k > 0) && (mod(k,freq) == 0)
            W = E + C*U;
            pobj = (mytrace(W,AB+U)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
            dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
            dfeas = max(0,svds(U,1)-mu);
            %        fprintf(' iter= %g  dobj= %g  dual feas= %g  pobj= %g  time= %g \n',k,dobj,dfeas,pobj,toc);
            fprintf('.');
            if abs(pobj-dobj) < tol*(abs(dobj)+1) && dfeas < tol
                fprintf('\n iter = %g  dobj = %g  dual feas = %g  pobj = %g  time = %g \n',k,dobj,dfeas,pobj,toc);
                stop = 1;
            end
        end
    end
    t_alg = toc;
    fprintf(' GP, no LS: iter/cpu/gap = %g/%g/%g\n',k,t_rankA + t_redA + t_CE + t_alg,abs(pobj-dobj)/(abs(dobj)+1))
end

%%%% Accelerated gradient-projection method I %%%%
if alg == 4
    k = 0;
    stop = 0;
    tic
    
    if init == 1
        U = zeros(n,m);		% initialize with U=0.
    else
        U = -AB;
        if init == 2
            [R,D,S] = svd(U,'econ');
            U = R*min(D,mu)*S';
        else
            svmax = svds(U,1);
            if svmax > mu
                U = U*(mu/svmax);	% initialize with U corresp. to W=0, scaled so its sv <= mu.
            end
        end
    end
    
    U0 = U;
    theta = 1;
    theta0 = 1;
    options.disp = 0;   % suppress display in eigs
    %  L=norm(C);
    L = eigs(C,1,'LM',options);			% works fine in practice
    fprintf(' L = eig(C) = %g\n', L);
    Wavg = zeros(n,m);		% initialize Wavg to zero.
    
    while stop == 0 && k <= maxiter
        Y = U + (theta/theta0-theta)*(U-U0);
        G = C*Y + E;
        [R,D,S] = svd(Y-G/L,'econ');    % For m<=n, R and D are m by m, S is n by m;
        % otherwise, S and D are n by n, R is m by n.
        U0 = U;
        U = R*min(D,mu)*S';
        Wavg = (1-theta)*Wavg + theta*G;
        
        % early termination if U-U0 is tiny
        
        if norm(U-U0,'fro') < meps
            fprintf(' termination due to negligible change in U = %g\n',norm(U-U0,'fro'));
            W = E + C*U;
            pobj = (mytrace(W,AB+U)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
            pobjavg = (mytrace(Wavg,M*Wavg)+m3)/2 - mytrace(AB,Wavg) + mu*sum(svd(Wavg));
            dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
            dfeas = max(0,svds(U,1)-mu);
            fprintf(' iter = %g  dobj = %g  dual feas = %g  pobj = %g  pobjavg = %g  time = %g \n',k,dobj,dfeas,pobj,pobjavg,toc);
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
            %        fprintf(' iter= %g  dobj= %g  dual feas= %g  pobj= %g  pobjavg= %g  time= %g\n',k,dobj,dfeas,pobj,pobjavg,toc);
            fprintf('.');
            if abs(min(pobj,pobjavg)-dobj) < tol*(abs(dobj)+1) && dfeas < tol
                fprintf('\n iter = %g  dobj = %g  dual feas = %g  pobj = %g  pobjavg = %g  time = %g\n',k,dobj,dfeas,pobj,pobjavg,toc);
                stop = 1;
            end
        end
    end
    t_alg = toc;
    fprintf(' AGP I: iter/cpu/gap = %g/%g/%g\n',k,t_rankA + t_redA + t_CE + t_alg,abs(min(pobj,pobjavg)-dobj)/(abs(dobj)+1))
    if min(pobj,pobjavg) == pobjavg
        W = Wavg;
        pobj = pobjavg;
    end
end

%%%% Accelerated gradient-projection method II %%%%
if alg == 5
    k = 0;
    stop = 0;
    tic
    
    if init == 1
        U = zeros(n,m);		% initialize with U=0.
    else
        U = -AB;
        if init == 2
            [R,D,S] = svd(U,'econ');
            U = R*min(D,mu)*S';
        else
            svmax = svds(U,1);
            if svmax > mu
                U = U*(mu/svmax);	% initialize with U corresp. to W=0, scaled so its sv <= mu.
            end
        end
    end
    
    Z = U;
    theta = 1;
    options.disp = 0;   % suppress display in eigs
    %  L = norm(C);
    L = eigs(C,1,'LM',options);			% works fine in practice
    fprintf(' L = eig(C) = %g\n', L);
    Wavg = zeros(n,m);		% initialize Wavg to zero.
    
    while stop == 0 && k <= maxiter
        
        Y = (1-theta)*U + theta*Z;
        G = C*Y + E;
        [R,D,S] = svd(Z-G/(theta*L),'econ');    % For m<=n, R and D are m by m, S is n by m;
        % otherwise, S and D are n by n, R is m by n.
        Z = R*min(D,mu)*S';
        U0 = U;
        U = (1-theta)*U + theta*Z;
        Wavg = (1-theta)*Wavg + theta*G;
        
        % early termination if U-U0 is tiny
        
        if norm(U-U0,'fro') < meps
            fprintf(' termination due to negligible change in U = %g\n',norm(U-U0,'fro'));
            W = E + C*U;
            pobj = (mytrace(W,M*W)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
            pobjavg = (mytrace(Wavg,M*Wavg)+m3)/2 - mytrace(AB,Wavg) + mu*sum(svd(Wavg));
            dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
            dfeas = max(0,svds(U,1)-mu);
            fprintf(' iter = %g  dobj = %g  dual feas = %g  pobj = %g  pobjavg = %g  time = %g \n',k,dobj,dfeas,pobj,pobjavg,toc);
            break
        end
        
        theta = 2*theta/(sqrt(theta^2+4)+theta);
        k = k + 1;
        if (k > 0) && (mod(k,freq) == 0)
            W = E + C*U;
            pobj = (mytrace(W,M*W)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
            pobjavg = (mytrace(Wavg,M*Wavg)+m3)/2 - mytrace(AB,Wavg) + mu*sum(svd(Wavg));
            dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
            dfeas = max(0,svds(U,1)-mu);
            %        fprintf(' iter = %g  dobj = %g  dual feas = %g  pobj = %g  pobjavg = %g  time = %g\n',k,dobj,dfeas,pobj,pobjavg,toc);
            fprintf('.');
            if abs(min(pobj,pobjavg)-dobj) < tol*(abs(dobj)+1) && dfeas < tol
                fprintf('\n iter = %g  dobj = %g  dual feas = %g  pobj = %g  pobjavg = %g  time = %g\n',k,dobj,dfeas,pobj,pobjavg,toc);
                stop=1;
            end
        end
    end
    t_alg = toc;
    fprintf(' AGP II: iter/cpu/gap = %g/%g/%g\n',k,t_rankA+t_redA+t_CE+t_alg,abs(min(pobj,pobjavg)-dobj)/(abs(dobj)+1))
    if min(pobj,pobjavg) == pobjavg
        W = Wavg;
        pobj = pobjavg;
    end
end

%%%% Accelerated gradient-projection method III %%%%
if alg == 6
    k = 0;
    stop = 0;
    tic
    
    Z = zeros(n,m);
    U = Z;
    theta = 1;
    Gavg = zeros(n,m);
    options.disp = 0;   % suppress display in eigs
    %  L = norm(C);
    L = eigs(C,1,'LM',options);			% works fine in practice
    fprintf(' L = eig(C) = %g\n', L);
    Wavg = zeros(n,m);		% initialize Wavg to zero.
    
    while stop == 0 && k <= maxiter
        Y = (1-theta)*U + theta*Z;
        G = C*Y + E;
        Gavg = Gavg + G/theta;
        [R,D,S] = svd(-Gavg/L,'econ');    % For m<=n, R and D are m by m, S is n by m;
        % otherwise, S and D are n by n, R is m by n.
        Z = R*min(D,mu)*S';
        U0 = U;
        U = (1-theta)*U + theta*Z;
        Wavg = (1-theta)*Wavg + theta*G;
        
        % early termination if U-U0 is tiny
        
        if norm(U-U0,'fro') < meps
            fprintf(' termination due to negligible change in U = %g\n',norm(U-U0,'fro'));
            W = E + C*U;
            pobj = (mytrace(W,M*W)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
            pobjavg = (mytrace(Wavg,M*Wavg)+m3)/2 - mytrace(AB,Wavg) + mu*sum(svd(Wavg));
            dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
            dfeas = max(0,svds(U,1)-mu);
            fprintf(' iter = %g  dobj = %g  dual feas = %g  pobj = %g  pobjavg = %g  time = %g \n',k,dobj,dfeas,pobj,pobjavg,toc);
            break
        end
        
        theta = 2*theta/(sqrt(theta^2+4)+theta);
        k = k + 1;
        if (k > 0) && (mod(k,freq) == 0)
            W = E + C*U;
            pobj = (mytrace(W,M*W)+m3)/2 - mytrace(AB,W) + mu*sum(svd(W));
            pobjavg = (mytrace(Wavg,M*Wavg)+m3)/2 - mytrace(AB,Wavg) + mu*sum(svd(Wavg));
            dobj = -mytrace(C*U,U)/2 - mytrace(E,U) - f/2;
            dfeas = max(0,svds(U,1)-mu);
            %        fprintf(' iter= %g  dobj= %g  dual feas= %g  pobj= %g  pobjavg= %g  time = %g\n',k,dobj,dfeas,pobj,pobjavg,toc);
            fprintf('.');
            if abs(min(pobj,pobjavg)-dobj) < tol*(abs(dobj)+1) && dfeas < tol
                fprintf('\n iter = %g  dobj = %g  dual feas = %g  pobj = %g  pobjavg = %g  time = %g\n',k,dobj,dfeas,pobj,pobjavg,toc);
                stop = 1;
            end
        end
    end
    t_alg = toc;
    fprintf(' AGP III: iter/cpu/gap = %g/%g/%g\n',k,t_rankA + t_redA + t_CE + t_alg,  abs(min(pobj,pobjavg)-dobj)/(abs(dobj)+1))
    if min(pobj,pobjavg) == pobjavg
        W = Wavg;
        pobj = pobjavg;
    end
end

if reduce_flag == 1
    W = R0*[W;zeros(nold-n,m)];
end


