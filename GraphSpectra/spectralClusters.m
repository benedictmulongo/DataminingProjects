
function []= spectralClusters() 
% rng(1)
E = csvread('example1.dat');
K = 5;
[A, number_of_nodes] = convertToAdjencyMatrix(E);
% [A, number_of_nodes] = convertToSimilarityMatrix(E, 4);
% Viz sparsity pattern
spy(A,'r'); 
% Viz of the graph
G = graph(A,'omitselfloops');
figure;
plot(G);
figure;
hold on;
p = plot(G);
axis equal

%Diagonal matrix
D = diag(sum(A,2));          
L = D^(-1/2)*A*D^(-1/2);
sizz = size(L);
% Laplacian = D-A;
Laplacian = D-A;
Laplacian = eye(sizz(1)) - L;

%Normalization
[eigVecsK, ~] = eigs(L,K,'la');
denom  =(sum(eigVecsK.^2,2)).^(1/2);
Y = bsxfun(@rdivide,eigVecsK,denom);

% Performs Kmeans
[idx,C] = kmeans(Y,K,'MaxIter',100);

% Highlight the clusters 
highlight(p,find(idx==1),'NodeColor','r')
highlight(p,find(idx==2),'NodeColor','g')
highlight(p,find(idx==3),'NodeColor','b')
highlight(p,find(idx==4),'NodeColor','c')
highlight(p,find(idx==5),'NodeColor','k')

% Viz Sorted Fiedler Vector
[eigVecs,~] = eigs(Laplacian, K,'sa');
eigVec = eigVecs(:,K); %if k=1 this would be the fieldler vector.
sortedEigenVectors = sort(eigVec);
figure;
plot(sortedEigenVectors)
end 

function [A, number_of_nodes] = convertToAdjencyMatrix(Edges)
col1 = Edges(:,1);
col2 = Edges(:,2);
number_of_nodes = max(max(col1,col2));
% Create a matrix of size M number_of_nodes X number_of_nodes 
% where M(col1,col2) = 1
As= sparse(col1, col2, 1, number_of_nodes, number_of_nodes); 
A = full(As);
end

function [A, number_of_nodes] = convertToSimilarityMatrix(Edges, sigma)
col1 = Edges(:,1);
col2 = Edges(:,2);
number_of_nodes = max(max(col1,col2));
sizer = size(col1);
len = sizer(1);
a_ij = zeros(len,1);
for i = 1:len
    if col1(i) == col2(i)
        a_ij(i) = 0 ;
    else
        a_ij(i) = gaussianKernel(col1(i), col2(i) , sigma) ;
    end
end
% Create a matrix of size M number_of_nodes X number_of_nodes 
% where M(col1,col2) = 1
As = sparse(col1, col2, a_ij, number_of_nodes, number_of_nodes); 
A = full(As);
end

function [a] = gaussianKernel(x_i, x_j, sigma) 
a = exp(-(x_i-x_j)^2/(2*sigma^2));
end