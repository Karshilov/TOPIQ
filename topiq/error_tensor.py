import numpy as np
import math


class ErrorTensor:
    """Error tensor T = (mu, sig2, bias, var_err, vu, vc, cov_xe).

    Propagates compression error statistics through primitive operators.
    vu + vc = var_err, where vu is the uncorrelated component and vc is
    the spatially correlated component (governed by the correlation index alpha).
    """

    def __init__(self, mu, var, bias=0.0, var_err=0.0, vu=None, vc=None,
                 cov_xe=0.0, eps=1e-12):
        self.mu = float(mu)
        self.sig2 = float(var)
        self.bias = float(bias)
        self.var_err = float(var_err)
        self.cov_xe = float(cov_xe)
        self.EPS = float(eps)
        if vu is None or vc is None:
            self.vu = float(var_err)
            self.vc = 0.0
        else:
            self.vu = float(vu)
            self.vc = float(vc)

    @staticmethod
    def split_vu_vc(var_err, alpha_raw, n_block):
        v = float(var_err)
        a = float(alpha_raw)
        n = int(n_block)
        if n <= 1:
            return v, 0.0
        vc = ((a - 1.0) * v) / (n - 1.0)
        vc = max(0.0, min(vc, v))
        vu = v - vc
        return vu, vc

    @staticmethod
    def alpha_to_r(alpha_raw, n0):
        n0 = int(n0)
        if n0 <= 1:
            return 0.0
        r = (float(alpha_raw) - 1.0) / (n0 - 1.0)
        return max(0.0, min(r, 1.0))

    @staticmethod
    def r_to_alpha(r, n):
        n = int(n)
        if n <= 1:
            return 1.0
        return 1.0 + (n - 1.0) * float(r)

    def with_alpha_base(self, alpha_base, n_base, n_block):
        r = self.alpha_to_r(alpha_base, n_base)
        alpha_n = self.r_to_alpha(r, n_block)
        vu, vc = self.split_vu_vc(self.var_err, alpha_n, n_block)
        return ErrorTensor(
            self.mu, self.sig2, self.bias, self.var_err,
            vu=vu, vc=vc, cov_xe=self.cov_xe, eps=self.EPS
        )

    def with_alpha(self, alpha_raw, n_block):
        vu, vc = self.split_vu_vc(self.var_err, alpha_raw, n_block)
        return ErrorTensor(self.mu, self.sig2, self.bias, self.var_err,
                           vu=vu, vc=vc, cov_xe=self.cov_xe, eps=self.EPS)

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.mu))
        ds = s * (1.0 - s)
        dds = ds * (1.0 - 2.0 * s)
        new_mu = s + 0.5 * dds * self.sig2
        new_sig2 = (ds ** 2) * self.sig2
        new_bias = ds * self.bias + 0.5 * dds * self.var_err + dds * self.cov_xe
        new_vu = (ds ** 2) * self.vu
        new_vc = (ds ** 2) * self.vc
        new_var_err = new_vu + new_vc
        new_cov = ds * self.cov_xe
        return ErrorTensor(new_mu, new_sig2, new_bias, new_var_err,
                           vu=new_vu, vc=new_vc, cov_xe=new_cov, eps=self.EPS)

    def relu(self):
        if self.mu > np.sqrt(self.sig2):
            return ErrorTensor(self.mu, self.sig2, self.bias, self.var_err,
                               self.vu, self.vc, cov_xe=self.cov_xe, eps=self.EPS)
        elif self.mu < -np.sqrt(self.sig2):
            return ErrorTensor(0, 0, 0, 0, 0, 0, eps=self.EPS)
        sig = np.sqrt(self.sig2) if self.sig2 > 1e-12 else 1e-12
        z = self.mu / sig
        Phi_z = 0.5 * (1.0 + math.erf(z / np.sqrt(2.0)))
        phi_z = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z ** 2)
        new_mu = self.mu * Phi_z + sig * phi_z
        new_sig2 = (self.sig2 + self.mu ** 2) * Phi_z + self.mu * sig * phi_z - new_mu ** 2
        ed = Phi_z
        bias_shift = phi_z * (self.var_err / (2.0 * sig))
        new_bias = ed * self.bias + bias_shift
        new_vu = (ed ** 2) * self.vu
        new_vc = (ed ** 2) * self.vc
        new_var_err = new_vu + new_vc
        new_cov = ed * self.cov_xe
        return ErrorTensor(new_mu, new_sig2, new_bias, new_var_err,
                           vu=new_vu, vc=new_vc, cov_xe=new_cov, eps=self.EPS)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return ErrorTensor(self.mu + float(other), self.sig2,
                               self.bias, self.var_err,
                               self.vu, self.vc, cov_xe=self.cov_xe, eps=self.EPS)
        return ErrorTensor(
            self.mu + other.mu,
            self.sig2 + other.sig2,
            self.bias + other.bias,
            self.var_err + other.var_err,
            self.vu + other.vu,
            self.vc + other.vc,
            cov_xe=self.cov_xe + other.cov_xe,
            eps=min(self.EPS, other.EPS)
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return ErrorTensor(self.mu - float(other), self.sig2,
                               self.bias, self.var_err,
                               self.vu, self.vc, cov_xe=self.cov_xe, eps=self.EPS)
        return ErrorTensor(
            self.mu - other.mu,
            self.sig2 + other.sig2,
            self.bias - other.bias,
            self.var_err + other.var_err,
            self.vu + other.vu,
            self.vc + other.vc,
            cov_xe=self.cov_xe - other.cov_xe,
            eps=min(self.EPS, other.EPS)
        )

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return ErrorTensor(float(other) - self.mu, self.sig2,
                               -self.bias, self.var_err,
                               self.vu, self.vc, cov_xe=-self.cov_xe, eps=self.EPS)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            k = float(other)
            kk = k * k
            return ErrorTensor(
                self.mu * k, self.sig2 * kk,
                self.bias * k, self.var_err * kk,
                self.vu * kk, self.vc * kk,
                cov_xe=self.cov_xe * k, eps=self.EPS
            )
        new_mu = self.mu * other.mu
        new_sig2 = (self.mu ** 2 * other.sig2 + other.mu ** 2 * self.sig2
                    + self.sig2 * other.sig2)
        a = other.mu
        b = self.mu
        new_bias = a * self.bias + b * other.bias
        ca = a * a + other.sig2
        cb = b * b + self.sig2
        new_vu = ca * self.vu + cb * other.vu
        new_vc = ca * self.vc + cb * other.vc
        new_var_err = new_vu + new_vc
        new_cov = self.mu * other.cov_xe + other.mu * self.cov_xe
        return ErrorTensor(new_mu, new_sig2, new_bias, new_var_err,
                           new_vu, new_vc, cov_xe=new_cov,
                           eps=min(self.EPS, other.EPS))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(1.0 / float(other))
        return self.__mul__(other.reciprocal())

    def reciprocal(self):
        if abs(self.mu) < 1e-9:
            return ErrorTensor(0, 0, 0, 0, 0, 0, self.EPS)
        mu = self.mu
        inv_mu = 1.0 / mu
        deriv = -inv_mu ** 2
        new_mu = inv_mu
        new_sig2 = (deriv ** 2) * self.sig2
        hessian = 2.0 * (inv_mu ** 3)
        new_bias = deriv * self.bias + 0.5 * hessian * self.var_err + hessian * self.cov_xe
        new_vu = (deriv ** 2) * self.vu
        new_vc = (deriv ** 2) * self.vc
        new_var_err = new_vu + new_vc
        return ErrorTensor(new_mu, new_sig2, new_bias, new_var_err,
                           vu=new_vu, vc=new_vc,
                           cov_xe=self.cov_xe * deriv, eps=self.EPS)

    def sum(self, n):
        n = int(n)
        new_bias = self.bias * n
        vu_sum = self.vu * n
        vc_sum = self.vc * (n * n)
        new_var_err = vu_sum + vc_sum
        new_cov = self.cov_xe * n
        return ErrorTensor(self.mu * n, self.sig2 * n, new_bias, new_var_err,
                           vu_sum, vc_sum, cov_xe=new_cov, eps=self.EPS)

    def mean(self, n):
        n = int(n)
        s = self.sum(n)
        k = 1.0 / n
        kk = k * k
        return ErrorTensor(s.mu * k, s.sig2 * kk, s.bias * k, s.var_err * kk,
                           vu=s.vu * kk, vc=s.vc * kk,
                           cov_xe=s.cov_xe * k, eps=self.EPS)

    def weighted_sum(self, sum_w, sum_w2):
        sw = float(sum_w)
        sw2 = float(sum_w2)
        new_bias = self.bias * sw
        new_var_err = self.vu * sw2 + self.vc * (sw * sw)
        new_cov = self.cov_xe * sw
        return ErrorTensor(np.nan, np.nan, new_bias, new_var_err,
                           vu=self.vu * sw2, vc=self.vc * (sw * sw),
                           cov_xe=new_cov, eps=self.EPS)

    def __pow__(self, k):
        k = int(k)
        if k != 2:
            raise NotImplementedError("Only k=2 is supported")
        mu = self.mu
        deriv = 2.0 * mu
        hessian = 2.0
        new_mu = mu * mu
        new_sig2 = (deriv ** 2) * self.sig2
        new_bias = deriv * self.bias + 0.5 * hessian * self.var_err + hessian * self.cov_xe
        Ex2 = mu * mu + self.sig2
        scale = 4.0 * Ex2
        new_vu = scale * self.vu
        new_vc = scale * self.vc
        new_var_err = new_vu + new_vc
        return ErrorTensor(new_mu, new_sig2, new_bias, new_var_err,
                           vu=new_vu, vc=new_vc,
                           cov_xe=self.cov_xe * deriv, eps=self.EPS)
