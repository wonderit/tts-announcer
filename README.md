### Setting

- Build monotonic alignment search
```sh
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
cd ..

PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files\eSpeak NG\libespeak-ng.dll
