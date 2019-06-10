function [e, x] = fALE(s, D, M, mu)
N = length(s);
u = zeros(M,N);
w = zeros(M,N);
x = zeros(1,N);

for n = D+M:length(s)
    
    for i = 1:M
        u(i,n) = s(n-D-i+1);            %Input vector
    end
    
    x(n) = w(:,n)'*u(:,n);              %Filter output
    e(n) = s(n) - x(n);                 %Prediction error
    w(:,n+1) = w(:,n) + mu*e(n)*u(:,n); %Weight update (LMS)
end
end