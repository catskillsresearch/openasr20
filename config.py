class Process:
    pass

C=Process()
C.stage='NIST'
C.language='amharic'
C.sample_rate=8000
C.recording='BABEL_OP3_307_82140_20140513_191321_inLine'
C.model_name=f'{C.language}_{C.sample_rate}_end2end_asr_pytorch_drop0.1_cnn_batch12_4_vgg_layer4'
C.model_dir=f'save/{C.model_name}'
C.best_model=f'{C.model_dir}/best_model.th'
C.batch_size=6
C.n_epochs=3
C.release='009'
C.grapheme_dictionary_fn = f'analysis/{C.language}/{C.language}_characters.json'
C.build_dir=f'{C.stage}/openasr20_{C.language}/build'
C.split_dir=f"{C.stage}/openasr20_{C.language}/build/audio_split"
C.nr_dir=f"{C.split_dir}_{C.sample_rate}"
