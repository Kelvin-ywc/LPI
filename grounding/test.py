import threading

def run1(event):
    print("开启线程1")
    for i in range(10):
        print('a' + str(i))
        event.wait()

def run2(event):
    print("开启线程2")
    for i in range(10):
        print('b' + str(i))
        event.wait()

event = threading.Event()
t1 = threading.Thread(target=run1, args=(event,))
t2 = threading.Thread(target=run2, args=(event,))

t1.start()
t2.start()

print("运行结束")
t1.join()
t2.join()

print("结束")