from RecordingTranscriptionSample import RecordingTranscriptionSample

def rts(task):
    (_config,x,y) = task
    return RecordingTranscriptionSample(_config,x,y)
