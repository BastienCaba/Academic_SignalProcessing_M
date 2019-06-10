function [e, x] = fANC(s, epsilon, mu, M)
N = length(s);      %Signal length
e = zeros(1, N);    %Error signal
x = zeros(1,N);     %Estimate signal
w = randn(M,N);     %Weights
u = zeros(M,N);     %Input signal

for n = M:N                             %Iterate over time
    for i = 1:M                         %Generate input u
        u(i, n) = epsilon(n-(i-1));    	%Filter input
    end
    x(n) = w(:,n)'*u(:,n);              %Predictor output
    e(n) = s(n) - x(n);                 %Error (filter output)
    w(:,n+1) = w(:,n) + mu*e(n)*u(:,n); %Weight update
end
end