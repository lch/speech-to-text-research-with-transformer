import jiwer


def analyse(reference, hypothesis):
    wer = jiwer.wer(reference, hypothesis)
    mer = jiwer.mer(reference, hypothesis)
    wil = jiwer.wil(reference, hypothesis)
    cer = jiwer.cer(reference, hypothesis)
    print(f'WER: {wer}')
    print(f'MER: {mer}')
    print(f'WIL: {wil}')
    print(f'CER: {cer}')