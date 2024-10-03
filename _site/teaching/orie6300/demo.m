%% This is the script for demonstrating the use of cvx, a Matlab-based 
% modeling system for convex optimization, see http://cvxr.com/cvx/.
% The main message of this tutorial: 
% CVX is extremely easy and intuitive to use!

%% Setup of cvx
% First go to http://cvxr.com/cvx/download/ and just downlaod the standard
% bundle with Gurobi and/or MOSEK. The following is a copy and paste from 
% Installation instruction section of the page http://cvxr.com/cvx/download/

% For most platforms, installation is relatively simple: 
% unpack the distribution to an empty directory, and then run cvx_setup
% from the MATLAB command line. Do not add CVX to your path yourself; 
% let cvx_setup do it for you. 
% Full instructions can be found in the Installation section 
% of the users� guide, found here online or included 
% with the distribution in the doc/ subdirectory.

% So just unpack the distribution in a empty directory and type cvx_setup 
% in your matlab command window!

% If you want to use other sovlers, i.e., Gurobi and MOSEK, you can go to 
% the installation instructin section and find the instruction there in
% using these two solvers. Basically, you can aquire free academic license 
% from the cvx team and then you are able to use the two solvers. 

%% What cvx is? 
%  I just want to mention cvx is just a modelling language for optimization
%  problems. This means that cvx itself is NOT solving the problem, i.e.,
%  itself is not an algorithm solving the problem you input. Instead, it is
%  an language that translates your input code to something the solver
%  understands. The solver here are SDPT3, gurobi, mosek, sedumi. Solvers
%  here mean algorithms that actually solves the optimization problem you
%  input. 

%% How large problem cvx can solve?
% Ignoring the problem structure, my past experiences tell me that it can
% solve a million variables with a few thousands constraints or 
% a million constraints with a few thousands variables. So it does 
% solve medium scale problems. You might just want to consider it as
% a tool for testing your ideas on some problems if you idea is
% not about designing algorithm solving the optimization problem. It also
% serve as the baseline for your algorithm if you claim your algorithm is
% scalable and fast. Indeed, if I can solver problem with cvx, why do I
% need your algorithm?

%% Getting started: Sparse vector Recovery 
% Since I work on statistics estimation problems all the days, let me
% introduce the sparse vector recovery problem. 

% Set up the sparse vector x0
rng(1)
n = 1000; %size of x0 
p = 0.01 ;%level of sparsity;
s = floor(n*p); % sparsity
r =randperm(n,s);
x0 = zeros(n,1);
x0(r) = randn(s,1);

% See how x0 looks like 
figure
plot(1:n,x0)
% Instead of observing x0 directly, usually we can only 
% measure x0 via a linear map A, suppose A is randomly generated 
% with number of rows (number of measurements) being s*log(s)*5
m = ceil(5*s*log(s));
A = randn(m,n);
y = A*x0;

% The conventional wisdom now is to minimize the l_1 norm subject to 
% measurments being matched as the signal 
% itself is sparse: rational is that the l_1 ball is the convex hull of 
% the sparse vectors. 
% We are trying to sovle 
%             minimize |x|_1 
%             subject to Ax = y

% How to input this into cvx ? 
cvx_begin 
variable x(n)
minimize norm(x,1)
subject to 
          A*x == y
cvx_end 
% the cvx_begin and cvx_end declare the cvx environment 
% variable x(n) declares x as a vector variable of length n 
% in the middle is the familar optimization problem 
% minimize xxx subject to xxx

% Let us see whether x and x0 are similar 
figure
plot(1:n,x0-x,'r-')
% The differnce is really small. 
norm(x0-x)/norm(x0)

%% The next problem we consider is an semidefinite program for 
% the Z2 synchronization problem: n entities are seperate to two groups: 
% 1 and -1. We may create a membership vector L with i-th entry indicating 
% the i-th membership, 1 if i belonging to group 1 and -1 if i belonging to
% group -1. 

% For each pair (i,j), i>=j, we observe the 1 +noise_ij  if entity i and entity j 
% are in the same group and -1+noise_ij  if they are in different group wiht certain noise.
% Write the previous observation in matrix, we observe C = J + E where J is
% the matrix with one and -one entries and E is the noise matrix. 
% We aim to find the matrix J which is rank 1. 

% The optimization problem 
% proposed is 
%            maximize trace( C*X)
%            subject to diag(X) == 1
%                       X is psd 

% set up number of entities and the size of first group, and membership
% vector l
n = 100; 
n1 = 50;
r =randperm(n,n1); % members in group -1
l = ones(n,1);
l(r) = -1;
s =find(l>0); % members in group 1
figure 
scatter(r,l(r),10,'r')
hold on 
scatter(s, l(s),10,'b')
legend('group -1','group 1')


% Set up noiseLevel and noise. The smaller the noise, the easier the problem
Noise = randn(n,n);
Noise = (Noise + Noise')/sqrt(2);
noiseLevel =sqrt(n/40);
% Set up C 
Obs = l*l'+ noiseLevel*Noise;
b = ones(n,1);
C = Obs;
figure
imagesc(l*l')
colorbar
figure
imagesc(C)
colorbar

% How to input the optimization problem into cvx ? 
cvx_begin 
variable X(n,n) semidefinite
maximize trace(C*X)
subject to 
       diag(X) == b
cvx_end 

% Note that we can declare the variable to be semidefinite
imagesc(X-l*l')
colorbar
norm(X-l*l')/norm(l*l')

%% The last problem (if time permits) we consider is an semidefinite program for 
% the matrix problem: We observe each entry of a low rank matrix Xnatural
% with probability p, can we recover Xnatural from the observation 

% Let k be the linear index of the observed entries, the optimization
% problem is 
% proposed is 
%            minimize nuclearnorm(X)
%            subject to X(k) == Xnatural(k)

rng(1) % Fixed random seed 
n = 100; 
r = 2;
p = 0.15;

%Generate the data, X is the underly true matrix Xnatural, 
% i,j are the matrix indecies of observed entries
% k is the linear indecices of observed entries
% b is the vector of observed entries 

w = sign(randn(n,r));
Xnatural = w*(w');
O = rand(n,n);
O = (O<=p);
Y = Xnatural.*O;
k = find(Y);
b = Xnatural(k);

figure
imagesc(abs(Y)>0)
colorbar
cvx_begin
variable X(n,n)
minimize norm_nuc(X)
subject to 
           X(k) == b
cvx_end 
norm(X-Xnatural)/norm(Xnatural)