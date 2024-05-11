import re, os
import requests
import json
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import urllib.request
urllib.request.urlretrieve("https://download.openxlab.org.cn/repos/file/Kevin676/NeuCoSVC-2/main?filepath=WavLM-Large.pt&sign=57fa9f151e6c9b9c9e1f784f0c64ecc5&nonce=1715420216902", "ckpt/WavLM-Large.pt")
urllib.request.urlretrieve("https://download.openxlab.org.cn/repos/file/Kevin676/NeuCoSVC-2/main?filepath=G_150k.pt&sign=a9c67e7da5e3c3b839a233a25972e01d&nonce=1715420243395", "ckpt/G_150k.pt")
urllib.request.urlretrieve("https://download.openxlab.org.cn/repos/file/Kevin676/NeuCoSVC-v2/main?filepath=speech_XXL_cond.zip&sign=4609f2f4d18fefb33b656cf462ef083f&nonce=1715420263779", "speech_XXL_cond.zip")
urllib.request.urlretrieve("https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/UVR-HP2.pth", "uvr5/uvr_model/UVR-HP2.pth")
urllib.request.urlretrieve("https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/UVR-HP5.pth", "uvr5/uvr_model/UVR-HP5.pth")

import zipfile
with zipfile.ZipFile("speech_XXL_cond.zip", 'r') as zip_ref:
    zip_ref.extractall("Phoneme_Hallucinator_v2/exp")

headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
}
pattern = r'//www\.bilibili\.com/video[^"]*'

def get_bilibili_video_id(url):
    match = re.search(r'/video/([a-zA-Z0-9]+)/', url)
    extracted_value = match.group(1)
    return extracted_value

# Get bilibili audio
def find_first_appearance_with_neighborhood(text, pattern):
    match = re.search(pattern, text)

    if match:
        return match.group()
    else:
        return None

def search_bilibili(keyword):
    if keyword.startswith("BV"):
      req = requests.get("https://search.bilibili.com/all?keyword={}&duration=1".format(keyword), headers=headers).text
    else:
      req = requests.get("https://search.bilibili.com/all?keyword={}&duration=1&tids=3&page=1".format(keyword), headers=headers).text

    video_link = "https:" + find_first_appearance_with_neighborhood(req, pattern)

    return video_link

def get_response(html_url):
  headers = {
      "referer": "https://www.bilibili.com/",
      "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
  }
  response = requests.get(html_url, headers=headers)
  return response

def get_video_info(html_url):
  response = get_response(html_url)
  html_data = re.findall('<script>window.__playinfo__=(.*?)</script>', response.text)[0]
  json_data = json.loads(html_data)
  if json_data['data']['dash']['audio'][0]['backupUrl']!=None:
    audio_url = json_data['data']['dash']['audio'][0]['backupUrl'][0]
  else:
    audio_url = json_data['data']['dash']['audio'][0]['baseUrl']
  video_url = json_data['data']['dash']['video'][0]['baseUrl']
  return audio_url, video_url

def save_audio(title, html_url):
  audio_url = get_video_info(html_url)[0]
  #video_url = get_video_info(html_url)[1]

  audio_content = get_response(audio_url).content
  #video_content = get_response(video_url).content

  with open(title + '.mp3', mode='wb') as f:
    f.write(audio_content)
  print("音乐内容保存完成")
  #with open(title + '.mp4', mode='wb') as f:
  #  f.write(video_content)
  #print("视频内容保存完成"

from uvr5.vr import AudioPre
weight_uvr5_root = "uvr5/uvr_model"
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

func = AudioPre
pre_fun_hp2 = func(
  agg=int(10),
  model_path=os.path.join(weight_uvr5_root, "UVR-HP2.pth"),
  device=device,
  is_half=True,
)

pre_fun_hp5 = func(
  agg=int(10),
  model_path=os.path.join(weight_uvr5_root, "UVR-HP5.pth"),
  device=device,
  is_half=True,
)

import webrtcvad
from pydub import AudioSegment
from pydub.utils import make_chunks

import os
import librosa
import soundfile
import gradio as gr


def vad(audio_name):
  audio = AudioSegment.from_file(audio_name, format="wav")
  # Set the desired sample rate (WebRTC VAD supports only 8000, 16000, 32000, or 48000 Hz)
  audio = audio.set_frame_rate(48000)
  # Set single channel (mono)
  audio = audio.set_channels(1)

  # Initialize VAD
  vad = webrtcvad.Vad()
  # Set aggressiveness mode (an integer between 0 and 3, 3 is the most aggressive)
  vad.set_mode(3)

  # Convert pydub audio to bytes
  frame_duration = 30  # Duration of a frame in ms
  frame_width = int(audio.frame_rate * frame_duration / 1000)  # width of a frame in samples
  frames = make_chunks(audio, frame_duration)

  # Perform voice activity detection
  voiced_frames = []
  for frame in frames:
      if len(frame.raw_data) < frame_width * 2:  # Ensure frame is correct length
          break
      is_speech = vad.is_speech(frame.raw_data, audio.frame_rate)
      if is_speech:
          voiced_frames.append(frame)

  # Combine voiced frames back to an audio segment
  voiced_audio = sum(voiced_frames, AudioSegment.silent(duration=0))

  voiced_audio.export("voiced_audio.wav", format="wav")




def youtube_downloader(
    video_identifier,
    filename,
    split_model,
    start_time
):
    print(video_identifier)
    video_info = get_video_info(video_identifier)[0]
    print(video_info)
    audio_content = get_response(video_info).content
    with open(filename.strip() + ".wav", mode="wb") as f:
        f.write(audio_content)
    audio_path = filename.strip() + ".wav"
    start_ms = start_time * 1000
    end_ms = start_ms + 45000
      # make dir output
    os.makedirs("output", exist_ok=True)

    if split_model=="UVR-HP2":
        pre_fun = pre_fun_hp2
    else:
        pre_fun = pre_fun_hp5

    audio_orig = AudioSegment.from_file(audio_path)
    if len(audio_orig) > end_ms:

      # Extract the segment
      segment = audio_orig[start_ms:end_ms]
      segment.export(filename.strip() + ".wav", format="wav")
      pre_fun._path_audio_(filename.strip() + ".wav", f"./output/{split_model}/{filename}/", f"./output/{split_model}/{filename}/", "wav")
      os.remove(filename.strip()+".wav")
    else:
      segment = audio_orig[start_ms:len(audio_orig)]
      segment.export(filename.strip() + ".wav", format="wav")
      pre_fun._path_audio_(filename.strip() + ".wav", f"./output/{split_model}/{filename}/", f"./output/{split_model}/{filename}/", "wav")
      os.remove(filename.strip()+".wav")


    return f"./output/{split_model}/{filename}/vocal_{filename}.wav_10.wav", f"./output/{split_model}/{filename}/instrument_{filename}.wav_10.wav"


def youtube_downloader_100s(
    video_identifier,
    filename,
    split_model
):
    print(video_identifier)
    video_info = get_video_info(video_identifier)[0]
    print(video_info)
    audio_content = get_response(video_info).content
    with open(filename.strip() + ".wav", mode="wb") as f:
        f.write(audio_content)
    audio_path = filename.strip() + ".wav"
    if split_model=="UVR-HP2":
        pre_fun = pre_fun_hp2
    else:
        pre_fun = pre_fun_hp5

    os.makedirs("output", exist_ok=True)
    audio_orig = AudioSegment.from_file(audio_path)

    if len(audio_orig) > 180000:
      start_ms = 30000
      end_ms = start_ms + 150000

      # Extract the segment

      segment = audio_orig[start_ms:end_ms]

      segment.export(filename.strip() + ".wav", format="wav")

      pre_fun._path_audio_(filename.strip() + ".wav", f"./output/{split_model}/{filename}/", f"./output/{split_model}/{filename}/", "wav")
      os.remove(filename.strip()+".wav")
    else:
      pre_fun._path_audio_(filename.strip() + ".wav", f"./output/{split_model}/{filename}/", f"./output/{split_model}/{filename}/", "wav")
      os.remove(filename.strip()+".wav")

    return f"./output/{split_model}/{filename}/vocal_{filename}.wav_10.wav", f"./output/{split_model}/{filename}/instrument_{filename}.wav_10.wav"


def convert(start_time, song_name_src, song_name_ref, ref_audio, check_song, auto_key, key_shift, vocal_vol, inst_vol):
  split_model = "UVR-HP5"
  #song_name_ref = song_name_ref.strip().replace(" ", "")
  #video_identifier = search_bilibili(song_name_ref)
  #song_id = get_bilibili_video_id(video_identifier)

  song_name_src = song_name_src.strip().replace(" ", "")
  video_identifier_src = search_bilibili(song_name_src)
  song_id_src = get_bilibili_video_id(video_identifier_src)

  if ref_audio is None:
      song_name_ref = song_name_ref.strip().replace(" ", "")
      video_identifier = search_bilibili(song_name_ref)
      song_id = get_bilibili_video_id(video_identifier)

      if os.path.isdir(f"./output/{split_model}/{song_id}")==False:
        audio, sr = librosa.load(youtube_downloader_100s(video_identifier, song_id, split_model)[0], sr=24000, mono=True)
        soundfile.write("audio_ref.wav", audio, sr)
      else:
        audio, sr = librosa.load(f"./output/{split_model}/{song_id}/vocal_{song_id}.wav_10.wav", sr=24000, mono=True)
        soundfile.write("audio_ref.wav", audio, sr)
    
      vad("audio_ref.wav")
  else:   
      vad(ref_audio)



  #if os.path.isdir(f"./output/{split_model}/{song_id_src}")==False:
  audio_src, sr_src = librosa.load(youtube_downloader(video_identifier_src, song_id_src, split_model, start_time)[0], sr=24000, mono=True)
  soundfile.write("audio_src.wav", audio_src, sr_src)
  #else:
  #  audio_src, sr_src = librosa.load(f"./output/{split_model}/{song_id_src}/vocal_{song_id_src}.wav_10.wav", sr=24000, mono=True)
  #  soundfile.write("audio_src.wav", audio_src, sr_src)
  if os.path.isfile("output_svc/NeuCoSVCv2.wav"):
    os.remove("output_svc/NeuCoSVCv2.wav")

  if check_song == True:
      if auto_key == True:
          os.system(f"python inference.py --src_wav_path audio_src.wav --ref_wav_path voiced_audio.wav")
      else:
          os.system(f"python inference.py --src_wav_path audio_src.wav --ref_wav_path voiced_audio.wav --key_shift {key_shift}")
 
  else:
      if auto_key == True:
          os.system(f"python inference.py --src_wav_path audio_src.wav --ref_wav_path voiced_audio.wav --speech_enroll")
      else:
          os.system(f"python inference.py --src_wav_path audio_src.wav --ref_wav_path voiced_audio.wav --key_shift {key_shift} --speech_enroll")
          
  audio_vocal = AudioSegment.from_file("output_svc/NeuCoSVCv2.wav", format="wav")

  # Load the second audio file
  audio_inst = AudioSegment.from_file(f"output/{split_model}/{song_id_src}/instrument_{song_id_src}.wav_10.wav", format="wav")

  audio_vocal = audio_vocal + vocal_vol  # Increase volume of the first audio by 5 dB
  audio_inst = audio_inst + inst_vol  # Decrease volume of the second audio by 5 dB

  # Concatenate audio files
  combined_audio = audio_vocal.overlay(audio_inst)

  # Export the concatenated audio to a new file
  combined_audio.export(f"{song_name_src}-AI翻唱.wav", format="wav")

  return f"{song_name_src}-AI翻唱.wav"



app = gr.Blocks()


with app:
  gr.Markdown("# <center>🥳💕🎶 NeuCoSVC v2 AI歌手全明星，无需训练、一键翻唱、重磅更新！</center>")
  gr.Markdown("## <center>🌟 只需 1 个歌曲名，一键翻唱任意歌手的任意歌曲，支持说话语音翻唱，随时随地，听你想听！</center>")
  gr.Markdown("### <center>🌊 [NeuCoSVC v2](https://github.com/thuhcsi/NeuCoSVC) 先享版 Powered by Tencent ARC Lab & Tsinghua University 💕</center>")
  with gr.Row():
    with gr.Column():
      with gr.Row():
        inp1 = gr.Textbox(label="请填写想要AI翻唱的歌曲或BV号", placeholder="七里香 周杰伦", info="直接填写BV号的得到的歌曲最匹配，也可以选择填写“歌曲名+歌手名”")
        inp2 = gr.Textbox(label="请填写含有目标音色的歌曲或BV号", placeholder="遇见 孙燕姿", info="例如您希望使用AI周杰伦的音色，就在此处填写周杰伦的任意一首歌")
      with gr.Row():
        inp0 = gr.Number(value=0, label="起始时间 (秒)", info="此程序将自动从起始时间开始提取45秒的翻唱歌曲")
        inp3 = gr.Checkbox(label="参考音频是否为歌曲演唱，默认为是", info="如果参考音频为正常说话语音，请取消打勾", value=True)
        inp4 = gr.Checkbox(label="是否自动预测歌曲人声升降调，默认为是", info="如果需要手动调节歌曲人声升降调，请取消打勾", value=True)
      with gr.Row():
        inp5 = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label="歌曲人声升降调", info="默认为0，+2为升高2个key，以此类推")
        inp6 = gr.Slider(minimum=-3, maximum=3, value=0, step=1, label="调节人声音量，默认为0")
        inp7 = gr.Slider(minimum=-3, maximum=3, value=0, step=1, label="调节伴奏音量，默认为0")
      btn = gr.Button("一键开启AI翻唱之旅吧💕", variant="primary")
    with gr.Column():
      ref_audio = gr.Audio(label="您也可以选择从本地上传一段音色参考音频。需要为去除伴奏后的音频，建议上传长度为60~90s左右的.wav文件；如果您希望通过歌曲名自动提取参考音频，请勿在此上传音频文件", type="filepath", interactive=True)
      out = gr.Audio(label="AI歌手为您倾情演唱的歌曲", type="filepath", interactive=False)

  btn.click(convert, [inp0, inp1, inp2, ref_audio, inp3, inp4, inp5, inp6, inp7], out)

  gr.Markdown("### <center>注意❗：请不要生成会对个人以及组织造成侵害的内容，此程序仅供科研、学习及个人娱乐使用。</center>")
  gr.HTML('''
      <div class="footer">
                  <p>🌊🏞️🎶 - 江水东流急，滔滔无尽声。 明·顾璘
                  </p>
      </div>
  ''')

app.queue().launch(show_error=True)
