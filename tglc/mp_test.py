from multiprocessing import get_start_method, set_start_method


print(get_start_method())
set_start_method("forkserver")
print(get_start_method())
