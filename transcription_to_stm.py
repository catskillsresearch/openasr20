if __name__=="__main__":
    from Cfg import Cfg
    from tqdm.auto import tqdm
    languages = ['amharic', 'pashto']
    languages = ['cantonese', 'guarani', 'javanese', 'kurmanji-kurdish', 'mongolian', 'somali', 'tamil', 'vietnamese']
    for language in languages:
       print('LANGUAGE', language)
       Cfg('NIST', 16000, language, 'build', '001').transcription_to_stm()

