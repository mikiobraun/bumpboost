import numpy as np
import pylab as pl
import data

class BumpBoost:
    """Boosting kernel bumps."""
    def __init__(self, iterations, width_candidates):
        self.iterations = iterations
        self.widths_candidates = width_candidates

    def fit(self, x, y):
        dims = x.shape[1]
        y2 = y.dot(y)
        self.alpha = np.zeros(self.iterations)
        self.width = np.zeros(self.iterations)
        self.base = np.zeros((self.iterations, dims))

        r = np.array(y)
        for n in range(self.iterations):
            i = self.choose(r)
            xi = x[i,:]
            c = np.zeros(len(self.widths_candidates))
            a = np.zeros(len(self.widths_candidates))
            for l, w in enumerate(self.widths_candidates):
                k = self.kernfct(w, xi, x)
                kr = k.dot(r)
                kk = k.dot(k)
                a[l] = kr / kk
                c[l] = a[l] * kr
            j = np.argmax(c)
            self.alpha[n] = a[j]
            self.width[n] = self.widths_candidates[j]
            self.base[n, :] = xi
            yh = self.alpha[n] * self.kernfct(self.width[n], xi, x)
            r -= yh
            print(f'iter #{n} residual {r.dot(r) / y2:.3f}% last alpha {self.alpha[n]}')

    def predict(self, x):
        result = np.zeros(len(x))
        for n in range(self.iterations):
            result += self.alpha[n] * self.kernfct(self.width[n], self.base[n,:], x)
        return result

    @staticmethod
    def choose(r):
        p = (r * r).cumsum()
        c = np.random.rand()
        return (p / p[-1] <= c).sum()

    @staticmethod
    def kernfct(tau, base, x):
        d = base - x
        return np.exp(-np.sum(np.power(10, -tau) * d * d, axis=1))

if __name__ == '__main__':
    x, y = data.sincdata(10000)

    def plotfit(x, y, m):
        xp = np.linspace(x.min(), x.max(),1000).reshape(-1, 1)
        yh = m.predict(xp)
        pl.plot(x, y, '.', xp, yh, '-')
        pl.show()

    #from sklearn.svm import SVR
    #svr = SVR()

    bb = BumpBoost(200, np.linspace(-2, 2))
    bb.fit(x,y)

    print(bb.alpha)

    plotfit(x, y, bb)


