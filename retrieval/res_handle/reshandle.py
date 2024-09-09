import json
import pandas as pd

def get_data(filename):
    data = None
    with open(filename,'r',encoding='utf8')as fp:
        data = json.load(fp)
    return data


class Eval():
    p1_tot, p5_tot, p10_tot = 0., 0., 0.
    last_p1, last_p5, last_p10 = 0., 0., 0,
    pre_p1_max, pre_p5_max, pre_p5_max = 0., 0., 0.
    cnt = 0

    def describe(self):
        s = ''
        forget = ''
        if self.cnt == 0:
            s = 'no element'
            forget = 'no forget'
        elif self.cnt == 1:
            s = f'P@1: {self.p1_tot / self.cnt}, P@5: {self.p5_tot / self.cnt}, P@10: {self.p10_tot / self.cnt}'
            forget = 'no forget'
        else:
            s = f'P@1: {self.p1_tot / self.cnt}, P@5: {self.p5_tot / self.cnt}, P@10: {self.p10_tot / self.cnt}'
            forget_1 = self.last_p1 - self.pre_p1_max
            forget_5 = self.last_p5 - self.pre_p5_max
            forget_10 = self.last_p10 - self.pre_p10_max
            forget = f'forget: P@1: {forget_1},  P@5: {forget_5},  P@10: {forget_10}, avg_forget: {(forget_1 + forget_5 + forget_10) / 3.0}'
        return s, forget

    def insert(self, data):
        self.cnt += 1
        self.p1_tot += data[0]
        self.p5_tot += data[1]
        self.p10_tot += data[2]
        if self.cnt == 1:
            self.pre_p1_max = data[0]
            self.pre_p5_max = data[1]
            self.pre_p10_max = data[2]
        else:
            self.pre_p1_max = max(self.last_p1, self.pre_p1_max)
            self.pre_p5_max = max(self.last_p5, self.pre_p5_max)
            self.pre_p10_max = max(self.last_p10, self.pre_p10_max)
        self.last_p1 = data[0]
        self.last_p5 = data[1]
        self.last_p10 = data[2]


class EvalList():
    # n = 12
    # eval_list = [Eval() for i in range(n)]
    # num_list = [73, 27, 44, 255, 210, 306, 474, 500, 500, 500, 500, 500]
    def __init__(self, n=12):
        self.n = n
        self.eval_list = [Eval() for i in range(n)]
        self.num_list = [73, 27, 44, 255, 210, 306, 474, 500, 500, 500, 500, 500]

    def insert(self, data, task_name='refcoco', task_type='val'):
        for i in range(self.n):
            for itm in data[str(i)][task_name][task_type]:
                # print(itm, data[str(i)]['refcoco']['val'][itm])
                self.eval_list[int(itm)].insert(data[str(i)][task_name][task_type][itm])

    def describe(self):
        p1, p5, p10 = 0., 0., 0.
        org_p1, org_p5, org_p10 = 0., 0., 0.
        forget_1, forget_5, forget_10 = 0., 0., 0.
        for i in range(self.n):
            # print(i, self.eval_list[i].cnt)
            p1 += self.eval_list[i].p1_tot * self.num_list[i] / self.eval_list[i].cnt
            p5 += self.eval_list[i].p5_tot * self.num_list[i] / self.eval_list[i].cnt
            p10 += self.eval_list[i].p10_tot * self.num_list[i] / self.eval_list[i].cnt
            org_p1 += self.eval_list[i].p1_tot / self.eval_list[i].cnt
            org_p5 += self.eval_list[i].p5_tot / self.eval_list[i].cnt
            org_p10 += self.eval_list[i].p10_tot / self.eval_list[i].cnt

            f_1 = self.eval_list[i].last_p1 - self.eval_list[i].pre_p1_max
            f_5 = self.eval_list[i].last_p5 - self.eval_list[i].pre_p5_max
            f_10 = self.eval_list[i].last_p10 - self.eval_list[i].pre_p10_max
            # print(f'task [{i}]: f1: last: {self.eval_list[i].last_p1}, max: {self.eval_list[i].pre_p1_max}')
            # print(f'task [{i}]: f5: last: {self.eval_list[i].last_p5}, max: {self.eval_list[i].pre_p5_max}')
            # print(f'task [{i}]: f10: last: {self.eval_list[i].last_p10}, max: {self.eval_list[i].pre_p10_max}')
            forget_1 += f_1
            forget_5 += f_5
            forget_10 += f_10
        total = sum(self.num_list)
        p1 /= total
        p5 /= total
        p10 /= total
        org_p1 /= self.n
        org_p5 /= self.n
        org_p10 /= self.n

        forget_1 /= (self.n - 1)
        forget_5 /= (self.n - 1)
        forget_10 /= (self.n - 1)
        print(f'org average precision: P@1: {org_p1}, P@5: {org_p5}, P@10: {org_p10}')
        # print(f'average precision: P@1: {p1}, P@5: {p5}, P@10: {p10}')
        print(
            f'average forget: P1: {forget_1}, P5: {forget_5}, P10:{forget_10}, avg forget: {(forget_1 + forget_5 + forget_10) / 3.0}')

    def describe_precision_detail(self):
        for i in range(self.n):
            s, forget = self.eval_list[i].describe()
            print(f'task {i}: {s}')

    def describe_forget_detail(self):
        for i in range(self.n):
            s, forget = self.eval_list[i].describe()
            print(f'task {i}: {forget}')

def get_res(filename,task_name='refcoco',task_type='val',n=12):
    data = get_data(filename)
    evalList = EvalList(n=n)
    evalList.insert(data, task_name=task_name, task_type=task_type)
    evalList.describe()
    evalList.describe_precision_detail()
    # evalList.describe_forget_detail()

# filename = './res/2024-04-06 04:46:40.263182.json'
# filename = './res/2024-04-06 06:04:17.874950.json'
# filename = './res/2024-04-06 08:55:43.259643.json'
# filename = './res/2024-04-06 18:17:41.105289.json'
# filename = 'res/2024-04-06 19:38:46.414162_hpi_best.json'
# filename = 'res/2024-04-06 21:15:55.409857_bestbest.json'
# filename = './res/2024-04-06 22:48:35.043734.json'
# filename = './res/2024-04-07 00:03:12.316331.json'
# filename = './res/2024-04-07 02:26:33.904797.json'
# filename = './res/2024-04-07 03:47:05.684449.json'
# filename = './res/2024-04-07 05:00:32.141429.json'
# filename = './res/2024-04-07 06:46:07.752224.json'
# filename = './res/2024-04-07 08:13:20.975115.json'
# filename = './res/2024-04-07 09:34:23.996457.json'
# filename = './res/2024-04-07 10:13:08.224830.json'
# filename = './res/2024-04-07 10:55:02.165937.json'
# filename = './res/2024-04-07 11:34:30.193744.json'
# filename = './res/2024-04-07 12:15:59.150264.json'
# filename = './res/2024-04-07 13:15:46.156653.json'
# filename = './res/2024-04-07 14:34:08.240848.json' # e5 init 0.02
# filename = './res/2024-04-07 14:11:01.302309.json'
# sprompt e10 0.02
filename = '../res/2024-04-07 16:49:45.568335.json'
get_res(filename, 'mscoco', 'i2t',12)

get_res(filename, 'mscoco', 't2i',12)
# # hpi e10
# filename = './res/2024-04-07 20:43:30.867178.json'
# get_res(filename, 'mscoco', 'i2t',12)
#
# get_res(filename, 'mscoco', 't2i',12)

# filename = './res/2024-04-07 22:09:13.556081.json'
# filename = './res/2024-04-08 01:14:14.189740.json'
filename = '../res/2024-04-09 20:13:34.397299.json'
get_res(filename, 'mscoco', 'i2t',12)

get_res(filename, 'mscoco', 't2i',12)
