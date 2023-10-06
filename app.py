import streamlit as st
import torch
import IPython.display as ipd
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

model_path_1 = 'models/dex'
model_path_2 = 'models/pej'
hps = None
    
st.set_page_config(page_title="TTS Mintalk 👄",
                  page_icon="📢",
                  initial_sidebar_state="expanded")

st.title("TTS Mintalk Artists 📢")

options = ["Dex", "Pyo"]
options_ko = ["덱스", "표은지"]
options2 = ["ko", "en"]


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def generate_voice(text, speaker):
    speaker_idx = options.index(speaker)
    symbol_lenth = 35
    if speaker_idx == 0:
        dir_path = model_path_1
    else:
        symbol_lenth = 207
        dir_path = model_path_2

    hps = utils.get_hparams_from_file(f"{dir_path}/config.json")

    net_g = SynthesizerTrn(
        symbol_lenth,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    _ = net_g.eval()
    _ = utils.load_checkpoint(f"{dir_path}/G.pth", net_g, None)

    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid = torch.LongTensor([speaker_idx])
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][
            0, 0].data.cpu().float().numpy()

    audio = ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False)
    st.audio(audio.data, format="audio/wav", start_time=0)

with open("design.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
speaker_box = st.selectbox("Select Artists",
                           options=options, index=1)

speaker_name = options_ko[options.index(speaker_box)]

st.subheader(f"아티스트 : {speaker_name} 님")

text = st.text_input("지문을 입력하세요.", f"안녕하세요, 저는 {speaker_name} 입니다.")

if st.button('Generate Voice'):
    generate_voice(text, speaker_box)