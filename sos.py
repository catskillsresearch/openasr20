def sos(task):
    artifact, goal = task
    return artifact.split_on_silence(goal_length_in_seconds=goal)
