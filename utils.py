from dataclasses import dataclass
from typing import Iterable, TypeAlias, Callable
from pathlib import Path
import time
import math
import hashlib
import json
import uuid

import requests
from torch import nn, Tensor
from tqdm import tqdm

from config import ARCH, TIME_DIM, TIMESTEP, DEVICE
from api_key import APP_KEY, APP_SECRET
from model import DDIM
#────────────────────────────────────────────────────────────────────
ImgLoader: TypeAlias = Iterable[Tensor]
#────────────────────────────────────────────────────────────────────
@dataclass
class WelfordStats:
    n: int = 0
    mean: Tensor|float = 0.0
    M2: Tensor|float = 0.0

    def calculate(self, loader: ImgLoader):
        """
        ```mean_AB = n*mean_A + B*mean_B
        = mean_A + (mean_B - mean_A) * B / (n+B)
        = mean_B + (mean_A - mean_B) * n / (n+B)
        = mean_A + (mean_B - mean_AB) * B / n
        = mean_B + (mean_A - mean_AB) * n / B
        ```
        """
        
        for x in tqdm(loader, "online calculate"):
            B = x.size(0)
            x = x.transpose(0,1).contiguous().flatten(1)
            mean_B = x.mean(1)
            m2_B = (B-1) * x.var(1)

            new_n = self.n + B
            delta = mean_B - self.mean
            self.mean += delta * B / new_n  # update mean
            self.M2 += m2_B + (delta * delta) * self.n * B / new_n  # update m2
            self.n = new_n  # update counter
            
    @property
    def variance(self):
        if self.n < 2:
            return float('nan')
        return self.M2 / (self.n - 1)
    
    @property
    def pvariance(self):
        if self.n < 2:
            return float('nan')
        return self.M2 / self.n

    @property
    def stdev(self):
        return self.variance.sqrt()
    
    @property
    def pstdev(self):
        return self.pvariance.sqrt()


@dataclass
class EMA:
    t: int = 0
    decay: float = 0.999  # value decay
    decay_: float = 0.6   # delta decay
    value: float = 0.0
    delta: float = 0.0    # used to calculate the variance of the value
    best: tuple[float, int] = (float('inf'), -1)  # the best value during training, used for reference only
    lerp: Callable[[float, float, float], float] = lambda a,b,w: a + w*(b-a)

    def update(self, x: float):
        assert not math.isnan(x), "EMA WARNNING: Get a NAN value!"
        
        if self.t == 0:
            self.value = x
            self.delta = 0.0

        self.value = self.lerp(self.value, x, 1-self.decay)
        self.delta = self.lerp(self.delta, (self.value - x)**2, 1-self.decay_)
        self.best = (min(self.best, self.value), self.t)  # update the best value and its time step
        self.t += 1

    def reset(self):
        self.t = 0  # the value and delta will be reset when t is 0
        self.best = (float('inf'), -1)
    
    @property
    def stdev(self):
        return self.delta**0.5


def summary(model: nn.Module) -> tuple[int, int]:
    MB = 1024 * 1024
    n = sum(p.numel() for p in model.parameters())
    m = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"The number of parameters: {n}.".center(50, '-'))
    print(f"The size of parameters: {m/MB:.2f} MB.".center(50, '-'))
    return n, m


@dataclass
class YoudaoTranslator:
    """
    Youdao 翻译器的简单封装（使用有道开放接口 v3）。
    """

    YOUDAO_URL: str = "https://openapi.youdao.com/api"
    app_key: str = APP_KEY
    app_secret: str = APP_SECRET
    TEST = "The quick brown fox jumps over the lazy dog."

    @staticmethod
    def _encrypt(sign_str: str) -> str:
        h = hashlib.sha256()
        h.update(sign_str.encode('utf-8'))
        return h.hexdigest()

    @staticmethod
    def _truncate(q: str|None) -> str:
        s = len(str(q))
        return q if s <= 20 else ''.join([q[0:10], str(s), q[s - 10:]])

    def translate(self, q: str = TEST, src: str = 'en', tgt: str = 'zh-CHS') -> dict:
        """Translate text `q` from `src` to `tgt` using Youdao API.\\
        Returns the parsed JSON response as a dict.
        """
        data = {
            'q': q,
            'from': src,
            'to': tgt,
            'signType': 'v3',
            'curtime': str(int(time.time())),
            'appKey': self.app_key,
            'salt': str(uuid.uuid1()),
        }

        sign_str = ''.join([
            self.app_key,
            self._truncate(q),
            data['salt'],
            data['curtime'],
            self.app_secret,
        ])

        data['sign'] = self._encrypt(sign_str)

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(self.YOUDAO_URL, data=data, headers=headers)
        return json.loads(response.content.decode())

    # allow calling instance directly for convenience
    def __call__(self, q: str = TEST, src: str = 'en', tgt: str = 'zh-CHS') -> dict:
        return self.translate(q, src=src, tgt=tgt)


#────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    youdao = YoudaoTranslator()
    print(youdao()['translation'])