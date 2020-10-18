def listen_to_file(sample_id, pred=None, label=None, proba=None):
    # Load the audio waveform using librosa
    filepath = test_samples[sample_id]['audio_filepath']
    audio, sample_rate = librosa.load(filepath,
                                      offset = test_samples[sample_id]['offset'],
                                      duration = test_samples[sample_id]['duration'])


    if pred is not None and label is not None and proba is not None:
        print(f"filepath: {filepath}, Sample : {sample_id} Prediction : {pred} Label : {label} Confidence = {proba: 0.4f}")
    else:
        
        print(f"Sample : {sample_id}")

    return ipd.Audio(audio, rate=sample_rate)
