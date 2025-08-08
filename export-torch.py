#!/usr/bin/env python3

import coremltools as ct
import torch
import sys
import moshi.models
import numpy as np

class ToTrace(torch.nn.Module):
    def __init__(self, model):
        super(ToTrace, self).__init__()
        self.model = model

    def forward(self, xs: torch.Tensor, state: list[torch.Tensor]):
        new_state = []

        # Seanet Encoder
        a = self.model.encoder.recurrent_n_states()
        this_state = state[:a]
        state = state[a:]
        xs, that_state = self.model.encoder.recurrent_forward(xs, this_state)
        new_state += that_state

        ## Transformer
        a = self.model.encoder_transformer.recurrent_n_states()
        this_state = state[:a]
        state = state[a:]
        xs, that_state = self.model.encoder_transformer.recurrent_forward(xs, this_state)
        #print("Got xs", xs)
        #print("Got that_state", that_state)
        new_state += that_state

        # Downsample
        xs = self.model.downsample(xs[0])
        for x in new_state:
            if not isinstance(x, torch.Tensor):
                print("nip bip")
        return xs, new_state

    def recurrent_init_state(self):
        y = []
        y += self.model.encoder.recurrent_init_state()
        y += self.model.encoder_transformer.recurrent_init_state()
        return y

class ToTrace2(torch.nn.Module):
    def __init__(self, model):
        super(ToTrace2, self).__init__()
        self.model = model

    def forward(self, xs: torch.Tensor, state: list[torch.Tensor]):
        new_state = []

        # Up sample
        xs = self.model.upsample(xs)

        ## Transformer
        a = self.model.decoder_transformer.recurrent_n_states()
        this_state = state[:a]
        print("n_states", a, len(this_state))
        state = state[a:]
        xs, that_state = self.model.decoder_transformer.recurrent_forward(xs, this_state)
        xs = xs[0]
        new_state += that_state

        # Seanet Encoder
        a = self.model.decoder.recurrent_n_states()
        this_state = state[:a]
        state = state[a:]
        print("xs", xs)
        xs, that_state = self.model.decoder.recurrent_forward(xs, this_state)
        new_state += that_state

        return xs, new_state

    def recurrent_init_state(self):
        y = []
        y += self.model.decoder_transformer.recurrent_init_state()
        y += self.model.decoder.recurrent_init_state()
        return y

info = moshi.models.loaders.CheckpointInfo.from_hf_repo(
    "kyutai/stt-1b-en_fr",
)       
            
mimi = info.get_mimi()
mimi.eval()
audio_chunk = torch.zeros((1,1,1920))
decoder_input = torch.zeros((1,512,1))
toTrace = ToTrace(mimi)
toTrace.eval()
toTrace2 = ToTrace2(mimi)
toTrace2.eval()

init_state = toTrace.recurrent_init_state()
ys, new_state = toTrace(audio_chunk, init_state)
#for i in range(200):
for i in range(10):
    ys, new_state = toTrace(audio_chunk, new_state)
    a = mimi.quantizer.encode(ys)
    print("Quantizer output", a.shape, a)
    a = mimi.decode_latent(a)
    a = mimi._to_encoder_framerate(a)
    decoder_input = mimi.decoder_transformer(a)[0].detach()

    print("Dequantizer output", a.shape)
    print("----------------")
    print(new_state)
print("BABA")
init_state_decode = toTrace2.recurrent_init_state()
print("init_state_decode", init_state_decode)
ys, new_state = toTrace2(a, init_state_decode)
# Export encoder to torchscript
a = torch.jit.trace(toTrace, example_inputs = [audio_chunk, init_state], strict = False)
ys, new_state = a(audio_chunk, init_state)
torch.jit.save(a, "mimi-encoder.torchscript")

# Export encoder to Microsoft ONNX
# dynamo output is broken
#torch.onnx.export(toTrace, (audio_chunk, init_state), "mimi-encoder.onnx", dynamo=True)
torch.onnx.export(toTrace, (audio_chunk, init_state), "mimi-encoder.onnx")

ctInputs = []
ctState = {}
# Note: inputs aren't flattend, we have [input, state: list]
# While output is flattend, we have [input, state...]
ctOutputs = [ct.TensorType(name = 'y')]
# Export encoder to Apple CoreML
for (i,x) in enumerate(init_state):
    n = 'state_' + str(i)
    on = 'out_state_' + str(i)
    ctInputs.append(ct.TensorType(name = n, shape = x.shape))
    ctOutputs.append(ct.TensorType(name = on))
    ctState[n] = np.zeros(x.shape)
    print(n, x.shape)
b = ct.convert(a,
        convert_to='mlprogram',
        inputs = [ct.TensorType(name = 'x', shape = [1,1,1920]), ctInputs],
        outputs = ctOutputs,
)
b.save('mimi-encoder.mlpackage')

ctState['x'] = np.array(audio_chunk)
# Try CoreML inference
b.predict(ctState)

# Export decoder to torchscript
a = torch.jit.trace(toTrace2, example_inputs = [decoder_input, init_state_decode], strict = False)
ys, new_state = a(decoder_input, init_state_decode)
torch.jit.save(a, "mimi-decoder.torchscript")

# Export encoder to Microsoft ONNX
# dynamo output is broken
#torch.onnx.export(toTrace, (audio_chunk, init_state), "mimi-encoder.onnx", dynamo=True)
torch.onnx.export(toTrace2, (decoder_input, init_state_decode), "mimi-decoder.onnx")

ctInputs = []
ctState = {}
# Note: inputs aren't flattend, we have [input, state: list]
# While output is flattend, we have [input, state...]
ctOutputs = [ct.TensorType(name = 'y')]
# Export encoder to Apple CoreML
for (i,x) in enumerate(init_state_decode):
    n = 'state_' + str(i)
    on = 'out_state_' + str(i)
    ctInputs.append(ct.TensorType(name = n, shape = x.shape))
    ctOutputs.append(ct.TensorType(name = on))
    ctState[n] = np.zeros(x.shape)
    print(n, x.shape)
input_shape = ct.Shape(shape=(1, 512, ct.RangeDim(lower_bound=1, upper_bound=2, default=1)))
b = ct.convert(a,
        convert_to='mlprogram',
        inputs = [ct.TensorType(name = 'x', shape = input_shape), ctInputs],
        outputs = ctOutputs,
)
b.save('mimi-decoder.mlpackage')

ctState['x'] = np.array(decoder_input)
# Try CoreML inference
b.predict(ctState)
