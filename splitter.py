def splitter(task):
    artifact, goal = task
    try:
        return (True, artifact.split(goal))
    except:
        return (False, artifact)
