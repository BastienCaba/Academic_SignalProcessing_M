function [H_new, y_hat, e] = clms(mu, X, y, H)
y_hat = H'*X;
e = y - y_hat;
H_new = H + mu*conj(e)*X;
end