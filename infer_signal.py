def infer_signal(model, signal, data_layer, data_loader):
    data_layer.set_signal(signal)
    batch = next(iter(data_loader))
    audio_signal, audio_signal_len = batch
    audio_signal, audio_signal_len = audio_signal.to(model.device), audio_signal_len.to(model.device)
    logits = model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
    return logits
