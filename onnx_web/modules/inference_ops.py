import onnxruntime as rt
import numpy as np

from .tokenizer import SimpleTokenizer # COPIED FROM https://github.com/openai/CLIP
tokenizer = SimpleTokenizer()

sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_cpu_mem_arena=False
clip_visual = rt.InferenceSession("./modules/clip_visual.onnx", sess_options, providers=['CPUExecutionProvider'])
clip_textual = rt.InferenceSession("./modules/clip_textual.onnx", sess_options, providers=['CPUExecutionProvider'])


def get_image_features(images):       # batch size always 1
    features = clip_visual.run([], {'input':images})[0]
    features/=np.linalg.norm(features,axis=1).reshape(-1,1)
    return features


def get_text_features(text):
    texts=[text]
    context_length=77

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]

    result = np.zeros((len(all_tokens), context_length), dtype=np.int32)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
        result[i, :len(tokens)] = np.array(tokens)
        
    features = clip_textual.run([], {'input':result})[0]
    features/=np.linalg.norm(features,axis=1).reshape(-1,1)
    return features