# localTTS
Your local TTS model inference framework.

# Supported models:

## kokoro

https://github.com/hexgrad/kokoro.git

#### Add dependency package
```bash
pip install kokoro
```

#### Usage

```python
from localtts import KokoroTTS as TTS

model_path = './temp/kokoro-82M/kokoro-v1_0.pth'
config_path = './temp/kokoro-82M/config.json'
voice_path = './temp/kokoro-82M/voices/af_heart.pt'
output_path = './temp/result.wav'

tts = TTS(model_path=model_path, config_path=config_path, voice_path=voice_path)

text = '''
The sky above the port was the color of television, tuned to a dead channel.
'''

paths = tts.infer(text, output_path=output_path, return_paths=True)

print("✅ wav file output：")
for path in paths:
    print(f"  → {path}")
```

## To be added ...