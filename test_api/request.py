import multiprocessing
import requests
import time

url = "localhost/test"

def send_request(i):
    data = {}
    # data['text'] = "2021年餐饮情况"
    response = requests.post(url, json=data)

if __name__ == "__main__":
    num_processes = 200  # 并发数，可调
    iters = 50

    start_time = time.time()
    for _ in range(iters):
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(send_request, range(num_processes))

    end_time = time.time()
    print(f"time: {(end_time - start_time)/num_processes/iters:.4f} seconds")
