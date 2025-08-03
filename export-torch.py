#!/usr/bin/env python3

import coremltools as ct
import torch
import moshi.models
import numpy as np

class ToTrace(torch.nn.Module):
    def __init__(self, model):
        super(ToTrace, self).__init__()
        self.model = model

    def forward(self, xs: torch.Tensor, state: list[torch.Tensor]):
        emb, new_state = self.model.recurrent_forward(xs, state)
        return emb, new_state

    def recurrent_init_state(self):
        return self.model.recurrent_init_state()

info = moshi.models.loaders.CheckpointInfo.from_hf_repo(
    "kyutai/stt-1b-en_fr",
)       
            
mimi = info.get_mimi()
audio_chunk = torch.zeros((1,1,1920))
toTrace = ToTrace(mimi.encoder)

# Export to torchscript
init_state = toTrace.recurrent_init_state()
a = torch.jit.trace(toTrace, example_inputs = [audio_chunk, init_state])
torch.jit.save(a, "mimi-seanet-encoder.torchscript")

# Export to Microsoft ONNX
torch.onnx.export(toTrace, (audio_chunk, init_state), "mimi-seanet-encoder.onnx", dynamo=True)

ctInputs = []
ctState = {}
# Note: inputs aren't flattend, we have [input, state: list]
# While output is flattend, we have [input, state...]
ctOutputs = [ct.TensorType(name = 'y')]
# Export to Apple CoreML
for (i,x) in enumerate(init_state):
    n = 'state_' + str(i)
    on = 'out_state_' + str(i)
    ctInputs.append(ct.TensorType(name = n, shape = x.shape))
    ctOutputs.append(ct.TensorType(name = on))
    ctState[n] = np.zeros(x.shape)
b = ct.convert(a,
        convert_to='mlprogram',
        inputs = [ct.TensorType(name = 'x', shape = [1,1,1920]), ctInputs],
        outputs = ctOutputs,
)
b.save('mimi-seanet-encoder.mlpackage')

ctState['x'] = np.array(audio_chunk)
# Try CoreML inference
b.predict(ctState)
