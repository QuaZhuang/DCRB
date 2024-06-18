function C=setdiffplus(A,B)
C = [];
% 遍历 A 中的每个元素
for i = 1:numel(A)
    if ~ismember(A(i), B)
        C=[C;A(i)];
    end
end
