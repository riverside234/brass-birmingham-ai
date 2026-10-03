"""Small terminal-only progress reporter; no dataset/report schema changes."""
import time


class Progress:
    def __init__(self, label, total, unit='decisions'):
        self.label,self.total,self.unit=label,total,unit
        self.next=20;self.started=time.monotonic()
        print(f'{label}: starting ({total} {unit})',flush=True)

    def __call__(self, completed):
        while self.next<=100 and completed*100>=self.next*self.total:
            print(f'{self.label}: {self.next}% ({completed}/{self.total} {self.unit}; '
                  f'{time.monotonic()-self.started:.1f}s)',flush=True)
            self.next+=20
