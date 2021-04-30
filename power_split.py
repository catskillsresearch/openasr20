from Clip import Clip

def power_split(C, audio, max_duration):
    size=audio.shape[0]
    tasks=[Clip(C.sample_rate, audio, audio, 0, size, size)]
    solution=[]
    while len(tasks):
        task, tasks = tasks[0], tasks[1:]
        good, bad = task.split_pass(max_duration)
        solution.extend(good)
        tasks.extend(bad)
    solution=list(sorted(solution, key=lambda x: x.parent_start))
    max_size = max([x.seconds for x in solution])
    if max_size > max_duration:
        msg = f"ERROR: Oversize clip: {max_size} seconds > {max_duration}"
        raise ValueError(msg)
    else:
        return solution
