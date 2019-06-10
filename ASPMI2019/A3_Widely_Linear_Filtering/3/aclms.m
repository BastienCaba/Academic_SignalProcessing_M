function [H_new, G_new, y_hat, e] = aclms(mu, X, y, H, G)
y_hat = H'*X + G'*conj(X);
e = y - y_hat;
H_new = H + mu*conj(e)*X;
G_new = G + mu*conj(e)*conj(X);
end