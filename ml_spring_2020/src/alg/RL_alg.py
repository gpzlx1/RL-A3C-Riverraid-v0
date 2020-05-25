from collections import defaultdict
import os

class RL_alg(object):
    def __init__(self):
        self.team = []
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)

    def step(self):
        raise NotImplementedError

    def logkv_mean(self,key,value):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key], self.name2cnt[key]= oldval*cnt/(cnt+1) + value/(cnt+1), cnt+1

    def dumpkvs(self):
        for student_id in self.team:
            self.name2val['student_id'] = student_id
            if not os.path.exists('./score.csv'):
                f = open('./score.csv', 'a+t')
                for (i, k) in enumerate(self.name2val):
                    if i > 0:
                        f.write(',')
                    f.write(k)
                f.write('\n')
            else:
                f = open('./score.csv', 'a+t')
                f.write('\n')

            for (i, k) in enumerate(self.name2val):
                if i > 0:
                    f.write(',')
                v = self.name2val[k]
                if v is not None:
                    f.write(str(v))


