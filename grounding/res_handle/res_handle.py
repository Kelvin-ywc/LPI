import json
import pandas as pd

# def get_data(filename):
#     data = None
#     with open(filename,'r',encoding='utf8')as fp:
#         data = json.load(fp)
#     return data
#
#
# class Eval():
#     p1_tot, p5_tot, p10_tot = 0., 0., 0.
#     last_p1, last_p5, last_p10 = 0., 0., 0,
#     pre_p1_max, pre_p5_max, pre_p5_max = 0., 0., 0.
#     cnt = 0
#
#     def describe(self):
#         s = ''
#         forget = ''
#         if self.cnt == 0:
#             s = 'no element'
#             forget = 'no forget'
#         elif self.cnt == 1:
#             s = f'P@1: {self.p1_tot / self.cnt}, P@5: {self.p5_tot / self.cnt}, P@10: {self.p10_tot / self.cnt}'
#             forget = 'no forget'
#         else:
#             s = f'P@1: {self.p1_tot / self.cnt}, P@5: {self.p5_tot / self.cnt}, P@10: {self.p10_tot / self.cnt}'
#             forget_1 = self.last_p1 - self.pre_p1_max
#             forget_5 = self.last_p5 - self.pre_p5_max
#             forget_10 = self.last_p10 - self.pre_p10_max
#             forget = f'forget: P@1: {forget_1},  P@5: {forget_5},  P@10: {forget_10}, avg_forget: {(forget_1 + forget_5 + forget_10) / 3.0}'
#         return s, forget
#
#     def insert(self, data):
#         self.cnt += 1
#         self.p1_tot += data[0]
#         self.p5_tot += data[1]
#         self.p10_tot += data[2]
#         if self.cnt == 1:
#             self.pre_p1_max = 0.
#             self.pre_p5_max = 0.
#             self.pre_p10_max = 0.
#         else:
#             self.pre_p1_max = max(self.last_p1, self.pre_p1_max)
#             self.pre_p5_max = max(self.last_p5, self.pre_p5_max)
#             self.pre_p10_max = max(self.last_p10, self.pre_p10_max)
#         self.last_p1 = data[0]
#         self.last_p5 = data[1]
#         self.last_p10 = data[2]
#
#
# class EvalList():
#
#     def __init__(self, n=12):
#         self.n = n
#         self.eval_list = [Eval() for i in range(n)]
#         self.num_list = [73, 27, 44, 255, 210, 306, 474, 500, 500, 500, 500, 500]
#     def insert(self, data, task_name='refcoco', task_type='val'):
#         for i in range(self.n):
#             for itm in data[str(i)][task_name][task_type]:
#                 # print(itm, data[str(i)]['refcoco']['val'][itm])
#                 self.eval_list[int(itm)].insert(data[str(i)][task_name][task_type][itm])
#
#     def describe(self):
#         p1, p5, p10 = 0., 0., 0.
#         org_p1, org_p5, org_p10 = 0., 0., 0.
#         forget_1, forget_5, forget_10 = 0., 0., 0.
#         for i in range(self.n):
#             p1 += self.eval_list[i].p1_tot * self.num_list[i] / self.eval_list[i].cnt
#             p5 += self.eval_list[i].p5_tot * self.num_list[i] / self.eval_list[i].cnt
#             p10 += self.eval_list[i].p10_tot * self.num_list[i] / self.eval_list[i].cnt
#             org_p1 += self.eval_list[i].p1_tot / self.eval_list[i].cnt
#             org_p5 += self.eval_list[i].p5_tot / self.eval_list[i].cnt
#             org_p10 += self.eval_list[i].p10_tot / self.eval_list[i].cnt
#
#             f_1 = self.eval_list[i].last_p1 - self.eval_list[i].pre_p1_max
#             f_5 = self.eval_list[i].last_p5 - self.eval_list[i].pre_p5_max
#             f_10 = self.eval_list[i].last_p10 - self.eval_list[i].pre_p10_max
#             forget_1 += f_1
#             forget_5 += f_5
#             forget_10 += f_10
#         total = sum(self.num_list)
#         p1 /= total
#         p5 /= total
#         p10 /= total
#         org_p1 /= self.n
#         org_p5 /= self.n
#         org_p10 /= self.n
#
#         forget_1 /= self.n
#         forget_5 /= self.n
#         forget_10 /= self.n
#         print(f'org average precision: P@1: {org_p1}, P@5: {org_p5}, P@10: {org_p10}')
#         print(f'average precision: P@1: {p1}, P@5: {p5}, P@10: {p10}')
#         print(
#             f'average forget: P1: {forget_1}, P5: {forget_5}, P10:{forget_10}, avg forget: {(forget_1 + forget_5 + forget_10) / 3.0}')
#
#     def describe_precision_detail(self):
#         for i in range(self.n):
#             s, forget = self.eval_list[i].describe()
#             print(f'task {i}: {s}')
#
#     def describe_forget_detail(self):
#         for i in range(self.n):
#             s, forget = self.eval_list[i].describe()
#             print(forget)
#
# def get_res(filename, task_name='refcoco', task_type='val',n=12):
#     data = get_data(filename)
#     evalList = EvalList(n=n)
#     evalList.insert(data, task_name=task_name, task_type=task_type)
#     evalList.describe()
#     evalList.describe_precision_detail()
#     evalList.describe_forget_detail()
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
            p1 += self.eval_list[i].p1_tot * self.num_list[i] / self.eval_list[i].cnt
            p5 += self.eval_list[i].p5_tot * self.num_list[i] / self.eval_list[i].cnt
            p10 += self.eval_list[i].p10_tot * self.num_list[i] / self.eval_list[i].cnt
            org_p1 += self.eval_list[i].p1_tot / self.eval_list[i].cnt
            org_p5 += self.eval_list[i].p5_tot / self.eval_list[i].cnt
            org_p10 += self.eval_list[i].p10_tot / self.eval_list[i].cnt

            f_1 = self.eval_list[i].last_p1 - self.eval_list[i].pre_p1_max
            f_5 = self.eval_list[i].last_p5 - self.eval_list[i].pre_p5_max
            f_10 = self.eval_list[i].last_p10 - self.eval_list[i].pre_p10_max
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
        print(f'average precision: P@1: {p1}, P@5: {p5}, P@10: {p10}')
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
    evalList.describe_forget_detail()

def get_res_simple(filename,task_name='refcoco',task_type='val',n=12):
    data = get_data(filename)
    evalList = EvalList(n=n)
    evalList.insert(data, task_name=task_name, task_type=task_type)
    evalList.describe()
    evalList.describe_precision_detail()
    evalList.describe_forget_detail()
# layer task interact
# filename = '../FINAL_RES/res_2024-03-23-21:52:46.json'
# task 10 epoch
# filename = '../FINAL_RES/res_2024-03-24-18:2:18.json'
# get_res(filename)
# decompose 5 epoch
# filename = '../FINAL_RES/res_2024-03-24-20:11:25.json'
# layer task
# filename = '../FINAL_RES/res_2024-03-25-5:37:30.json'
# decompose 10 epoch
# filename = '../FINAL_RES/res_2024-03-25-9:22:26.json'

#  task BASE_LR: 0.05 MIN_LR: 0.01
# filename = '../FINAL_RES/res_2024-03-25-16:27:12.json'
# filename = '../FINAL_RES/res_2024-03-25-18:50:6.json'
# filename = '../FINAL_RES/res_2024-03-25-21:24:45.json'
# filename = '../FINAL_RES/res_2024-03-26-5:13:0.json'
# get_res(filename, task_type='val')
# get_res(filename, n=11, task_type='val')
# get_res(filename, task_type='testA')
# get_res(filename, task_type='testB')
# filename = '../FINAL_RES/res_2024-03-27-2:35:32.json'
# get_res(filename, task_name='refcoco+', task_type='val')
# get_res(filename, task_name='refcoco+', task_type='testA')
# get_res(filename, task_name='refcoco+', task_type='testB', n=11)
    
# filename = '../FINAL_RES/res_2024-03-27-22:14:12_layer.json'
# get_res(filename, task_name='refcoco', task_type='val')

# filename = '../FINAL_RES/res_2024-03-28-18:54:53.json'
# filename = '../FINAL_RES/res_2024-03-29-17:1:17.json'
# filename = '../FINAL_RES/res_2024-03-31-7:28:25.json'
# filename = '../FINAL_RES/res_2024-04-10-19:4:10.json'
# get_res(filename, task_name='refcoco', task_type='val')

filename = '../FINAL_RES/res_2024-04-14-21:28:41.json'
get_res(filename, task_name='refcoco', task_type='val')

filename = '../FINAL_RES/res_2024-04-15-2:36:29.json'
get_res(filename, task_name='refcoco', task_type='val')
