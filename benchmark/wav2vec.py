from transformers import pipeline
import torch

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

import os

default_model = 'wav2vec2-xlsr-multilingual-56'
default_model_dir = os.path.dirname(os.path.realpath(__file__)) + '/models'
default_models_path = f'{default_model_dir}/{default_model}'


class Wav2vec:
    def __init__(self,
                 chunk_length: int,
                 language: str,
                 model_path: str = default_models_path) -> None:
        self.task = "automatic-speech-recognition"
        self.language = language
        self.model_path = model_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.chunk_length = chunk_length

    def load_processor(self):
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = self.processor.feature_extractor

    def load_model(self):
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_path).to(
            self.device)

    def construct_pipeline(self):
        self.pipeline = pipeline(
            self.task,
            model=self.model,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            chunk_length_s=self.chunk_length,
            device=self.device,
        )

    def pre(self):
        self.load_processor()
        self.load_model()
        self.construct_pipeline()

    def regoconize(self, audio_array):
        if not hasattr(self, 'model'):
            print('Please load model first.')
            return
        if not hasattr(self, 'pipeline'):
            print('Please construct pipeline first.')
            return
        # audio_tensor = torch.from_numpy(audio_array).to(self.device)
        return self.pipeline(audio_array)
