"""Training-only permutation/Welch tests, declared-family BH and fixed ranking."""
import numpy as np
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Effect size requires >=2 finite subjects in each class")
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    if pooled == 0:
        return 0.0 if a.mean() == b.mean() else float(np.sign(a.mean()-b.mean()) * np.inf)
    return float((a.mean()-b.mean())/pooled)


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty(len(p))
    q[order] = np.minimum(1, np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    return q


class Selector:
    def __init__(self, families, alpha=0.05, correlation=0.70, method="permutation", permutations=5000, seed=42, caps=None, testing_families=None):
        self.families = np.asarray(families)
        self.method, self.permutations, self.seed = method, permutations, seed
        self.caps = caps or {}
        self.testing_families = np.asarray(testing_families if testing_families is not None else families)
        self.alpha = alpha
        self.correlation = correlation

    def fit(self, x, y):
        a, b = x[y == 1], x[y == 0]
        if min(len(a), len(b)) < 2:
            raise ValueError("Too few training subjects for feature selection")
        p = np.ones(x.shape[1])
        effect = np.zeros(x.shape[1])
        for j in range(x.shape[1]):
            if np.ptp(x[:, j]) == 0:
                continue
            effect[j] = abs(cohen_d(a[:, j], b[:, j]))
            if self.method == "welch":
                p[j] = 0 if np.isinf(effect[j]) else ttest_ind(a[:, j], b[:, j], equal_var=False).pvalue
        if self.method == "permutation":
            rng = np.random.default_rng(self.seed)
            observed = np.abs(a.mean(axis=0)-b.mean(axis=0))
            counts = np.zeros(x.shape[1],int)
            for start in range(0,self.permutations,128):
                size=min(128,self.permutations-start)
                shuffled=np.stack([rng.permutation(y) for _ in range(size)])
                weights=shuffled/len(a)-(1-shuffled)/len(b)
                diffs=np.abs(weights @ x)
                counts += (diffs >= observed-1e-12*np.maximum(1,observed)).sum(axis=0)
            p=(counts+1)/(self.permutations+1)
            p[np.ptp(x,axis=0)==0]=1.0
        elif self.method != "welch":
            raise ValueError("Unknown feature comparison method")
        if not np.isfinite(p).all():
            raise ValueError("Undefined feature test")
        q = np.ones(len(p))
        for family in np.unique(self.testing_families):
            ix = self.testing_families == family
            q[ix] = bh(p[ix])
        candidates = [j for j in np.argsort(-effect, kind="stable") if q[j] <= self.alpha]
        limited=[]
        for family in np.unique(self.families):
            group=[j for j in candidates if self.families[j]==family]
            cap=self.caps.get(str(family))
            limited.extend(group if cap is None else group[:int(cap)])
        candidates=[j for j in candidates if j in limited]
        keep = []
        for j in candidates:
            if all(abs(np.corrcoef(x[:, j], x[:, k])[0, 1]) <= self.correlation for k in keep):
                keep.append(j)
        self.indices = np.array(keep, int)
        self.p, self.q, self.effect = p, q, effect
        self.scaler = StandardScaler().fit(x[:, self.indices]) if keep else None
        return self

    def transform(self, x):
        return self.scaler.transform(x[:, self.indices]) if self.scaler else np.empty((len(x), 0))
