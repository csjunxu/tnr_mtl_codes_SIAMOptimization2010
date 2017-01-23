March, 2010
July, 2010, uploaded mytrace.m

This subdirectory contains the following Matlab source codes, which deals with minimization of
least square loss with nuclear norm regularization:

mat_dual.m          This provides a choice of 6 different algorithms with 3 different initialization
                    to solve the dual problem

mat_primal.m        This provides a choice of 3 different algorithms with 2 different initialization
                    to solve the primal problem

mat_primal_dual.m   This code solves the problem with different input of regularization parameters.
                    It chooses to solve the primal or dual problem depending on the regularization
                    parameter, and uses the solution for another parameter as warm start.

mytrace.m           A subroutine needed for all three codes.

The accelerated algorithm I, II, III are named as in the following preprint:
    Paul Tseng
    "Approximation accuracy, gradient methods, and error bound for structured convex optimization"
    submitted to Mathematical Programming, also available at
    http://www.math.washington.edu/~tseng/papers.html

Implementation and some numerical experience with the above codes are described in the paper: 
    Ting Kei Pong, Paul Tseng, Shuiwang Ji, and Jieping Ye
    "Trace norm regularization: reformulations, algorithms, and multi-task learning"
    submitted to SIAM Journal of Optimization 
The codes were last updated on June 16, 2009. They were all written jointly with Paul Tseng.
    
Questions/comments/suggestions about the codes are welcome.  

Ting Kei Pong, tkpong@math.washington.edu