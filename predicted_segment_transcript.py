import numpy as np
from transcribe import transcribe
from normalize import normalize
from smooth import smooth
from extremize import extremize
from mask_boundaries import mask_boundaries
from allocate_pred_to_speech_segments import allocate_pred_to_speech_segments

def predicted_segment_transcript(C, model, audio, start, end, s_dB_mean, samples_per_spect, dt_S):
    clip_audio=audio[start:end]
    prediction=transcribe(C, model, clip_audio)
    spec_start=int(start/samples_per_spect)
    spec_end=int(end/samples_per_spect)
    clip_power=s_dB_mean[spec_start:spec_end]
    normalized_power=normalize(np.copy(clip_power))
    timeline=np.arange(spec_start,spec_end)*dt_S
    w=30
    smoothed_normalized_power=smooth(normalized_power,w)
    speech_mask=extremize(smoothed_normalized_power, 0.2)
    speech_segments=mask_boundaries(speech_mask)+spec_start
    spec_to_words=allocate_pred_to_speech_segments(prediction, speech_segments)
    segment_transcript = [(spec1*dt_S, (spec2-spec1)*dt_S, word) for spec1, spec2, word in spec_to_words]
    return segment_transcript
