# rvclib

```bash
# install 
pip install git+https://github.com/eschmidbauer/rvclib.git

# download models
python -m 'rvclib.rvc.lib.tools.prerequisites_download'
```

```bash
# to run on mac arm64
export OMP_NUM_THREADS=1
```

```python
import torch
import librosa
from scipy.io import wavfile
import numpy as np
from rvclib.rvc.infer.infer import VoiceConverter
from rvclib.rvc.infer.pipeline import Pipeline as VC
from rvclib.rvc.lib.algorithm.synthesizers import Synthesizer
from rvclib.rvc.lib.utils import load_embedding


rvc_model_path: str = "voice.pth"
rvc_index_path: str = "voice.index"
input_wav: str = "input.wav"
output_wav: str = "output.wav"


infer_pipeline = VoiceConverter()
infer_pipeline.cpt = torch.load(rvc_model_path, map_location="cpu")
infer_pipeline.tgt_sr = infer_pipeline.cpt["config"][-1]
infer_pipeline.cpt["config"][-3] = infer_pipeline.cpt["weight"]["emb_g.weight"].shape[0]
infer_pipeline.use_f0 = infer_pipeline.cpt.get("f0", 1)
infer_pipeline.version = infer_pipeline.cpt.get("version", "v1")
infer_pipeline.text_enc_hidden_dim = 768 if infer_pipeline.version == "v2" else 256
infer_pipeline.net_g = Synthesizer(*infer_pipeline.cpt["config"], use_f0=infer_pipeline.use_f0, text_enc_hidden_dim=infer_pipeline.text_enc_hidden_dim, is_half=infer_pipeline.config.is_half)
del infer_pipeline.net_g.enc_q

infer_pipeline.net_g.load_state_dict(infer_pipeline.cpt["weight"], strict=False)
infer_pipeline.net_g.eval().to(infer_pipeline.config.device)
infer_pipeline.net_g = (infer_pipeline.net_g.half() if infer_pipeline.config.is_half else infer_pipeline.net_g.float())

infer_pipeline.vc = VC(infer_pipeline.tgt_sr, infer_pipeline.config)
infer_pipeline.n_spk = infer_pipeline.cpt["config"][-3]
infer_pipeline.loaded_model = rvc_model_path

infer_pipeline.hubert_model = load_embedding("contentvec")
infer_pipeline.hubert_model.to(infer_pipeline.config.device)
infer_pipeline.hubert_model = (infer_pipeline.hubert_model.half() if infer_pipeline.config.is_half else infer_pipeline.hubert_model.float())
infer_pipeline.hubert_model.eval()
infer_pipeline.last_embedder_model = "contentvec"

naudio, sr = librosa.load(input_wav)
naudio = librosa.resample(naudio, orig_sr=sr, target_sr=16000)
audio = np.array(naudio).flatten()
audio_max = np.abs(audio).max() / 0.95
if audio_max > 1:
    audio /= audio_max

audio_opt = infer_pipeline.vc.pipeline(
    model=infer_pipeline.hubert_model,
    net_g=infer_pipeline.net_g,
    sid=0,
    audio=audio,
    pitch=0,
    f0_method="rmvpe",
    file_index=rvc_index_path,
    index_rate=0.3,
    pitch_guidance=infer_pipeline.use_f0,
    filter_radius=3,
    volume_envelope=1,
    version=infer_pipeline.version,
    protect=0.33,
    hop_length=128,
    f0_autotune=False,
    f0_autotune_strength=1.0,
    f0_file=None,
)

wavfile.write(output_wav, infer_pipeline.tgt_sr, audio_opt)
```
