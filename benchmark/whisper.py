from transformers import pipeline
import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor

import os

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
# model = WhisperForConditionalGeneration.from_pretrained("./whisper-large-v2",
#                                                         low_cpu_mem_usage=True)
default_model = 'whisper-large-v2'
default_model_dir = os.path.dirname(os.path.realpath(__file__)) + '/models'
default_models_path = f'{default_model_dir}/{default_model}'


def model_download(model_dir=default_model_dir, model_name=default_model):
    model_path = f'{model_dir}/{model_name}'
    os.mkdir(model_path)


class Whisper:
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
        self.processor = WhisperProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = self.processor.feature_extractor

        # self.tokenizer = WhisperTokenizer.from_pretrained(
        #     self.model_path, language=self.language, task='transcribe')
        # self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
        #     self.model_path)

    def load_model(self, low_mem: bool):
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_path, low_cpu_mem_usage=low_mem)
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language, task="transcribe")

    def construct_pipeline(self):
        self.pipeline = pipeline(
            self.task,
            model=self.model,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            chunk_length_s=self.chunk_length,
            device=self.device,
        )

    def pre(self, low_mem=False):
        self.load_processor()
        self.load_model(low_mem=low_mem)
        self.construct_pipeline()

    def regoconize(self, audio_array):
        if not hasattr(self, 'model'):
            print('Please load model first.')
            return
        if not hasattr(self, 'pipeline'):
            print('Please construct pipeline first.')
            return

        return self.pipeline(audio_array)
