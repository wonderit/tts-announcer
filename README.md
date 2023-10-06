---
title: TTS-Mintalk
emoji: 📢
colorFrom: purple
colorTo: gray
sdk: streamlit
sdk_version: 1.27.0
app_file: app.py
pinned: true
---



### Setting
- python 3.10

- Build monotonic alignment search
```sh
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
cd ..

PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files\eSpeak NG\libespeak-ng.dll
```
- setup github repo and huggingface space together
```
git remote add space https://huggingface.co/spaces/wonderit-safeai/tts-mintalk
git push --force space main
```

