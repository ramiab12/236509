import torch
from time import time
import parallel_matmul_cuda
import matplotlib.pyplot as plt

torch.manual_seed(42)

N = []


def fill_N(exp):
    for i in range(0, exp):
        N.append(pow(2, i + 1))


def compute_time_for_a_given_setting(A, B, T, TB, iterations):
    A_cuda = A.to('cuda')
    B_cuda = B.to('cuda')
    start_time = time()
    for i in range(0, iterations):
        C = parallel_matmul_cuda.ParallelMatMul(A_cuda, B_cuda, T, TB)
    end_time = time()
    C = C.to('cpu')
    return C, end_time - start_time

def check_equal(A, B, C, n, tb = None):
    expected = torch.matmul(A, B)
    sub = torch.sub(expected, C)
    squared_err = torch.sum(torch.pow(sub, 2))
    if squared_err >= 10e-3:
        err_msg = "for n = %d" % n
        if tb:
            err_msg += ", tb = %d" % tb
        print("============================")
        print(err_msg)
        print("============================")
        print(expected)
        print(C)

def part_2_3(iterations, part, message):
    T = 256
    TB = 1
    run_times = []
    for n in N:
        A = torch.rand(n, n)
        if part == "a":
            B = torch.rand(n)
        else:
            B = torch.rand(n, n)
        torch.cuda.empty_cache()
        C, run_time = compute_time_for_a_given_setting(A, B, T, TB, iterations)
        check_equal(A, B, C, n)
        run_times.append(run_time)
    plt.close()
    plt.plot(N, run_times)
    plt.xlabel("n")
    plt.ylabel("Time for 100 iterations [s]")
    plt.title(message)
    plt.show()
    plt.savefig("Part1_" + str(part) + ".png")


def part_4_a_b(iterations, message, part):
    T = 256
    run_times = []
    run_times_per_tb = {}
    for tb in N:  # same range
        for n in N:
            A = torch.rand(n, n)
            if part == "a":
                B = torch.rand(n)
            else:
                B = torch.rand(n, n)
            torch.cuda.empty_cache()
            C, run_time = compute_time_for_a_given_setting(A, B, T, tb, iterations)
            run_times.append(run_time)
            check_equal(A, B, C, n, tb)
        run_times_per_tb[str(tb)] = run_times
        run_times = []
    _, axis = plt.subplots(4, 2, figsize=(15, 12))
    for i, tb in enumerate(N):
        plt.subplot(4, 2, i + 1)
        plt.plot(N, run_times_per_tb[str(tb)])
        plt.xlabel("n")
        plt.ylabel("Time for 100 iterations [s]")
        plt.title(message + " for TB = %d" % tb)
    plt.show()
    plt.savefig("Part2_" + str(part) + ".png")

if __name__ == "__main__":
    fill_N(8)
    part_2_3(100, "a", "Matrix x Vector")
    part_2_3(100, "b", "Matrix x Matrix")
    part_4_a_b(100, "Matrix x vector", "a")
    part_4_a_b(100, "Matrix x Matrix", "b")
