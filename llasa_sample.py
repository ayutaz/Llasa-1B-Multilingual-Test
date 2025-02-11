import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import soundfile as sf

# 環境変数からトークンを取得
token = os.environ.get("HF_HUB_TOKEN")

llasa_1b = 'HKUSTAudio/Llasa-1B-Multilingual'
# token 引数を利用して認証トークンを渡す
tokenizer = AutoTokenizer.from_pretrained(llasa_1b, token=token)
model = AutoModelForCausalLM.from_pretrained(llasa_1b, token=token)
model.eval() 
model.to('cuda')

from xcodec2.modeling_xcodec2 import XCodec2Model
model_path = "HKUSTAudio/xcodec2"
# 同様に token を渡す
Codec_model = XCodec2Model.from_pretrained(model_path, token=token)
Codec_model.eval().cuda()   

input_text = '言いなりにならなきゃいけないほど後ろめたい事をしたわけでしょ。'

def ids_to_speech_tokens(speech_ids):
    return [f"<|s_{speech_id}|>" for speech_id in speech_ids]

def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num = int(token_str[4:-2])
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

with torch.no_grad():
    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
    chat = [
        {"role": "user", "content": "Convert the text to speech:" + formatted_text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
    ]
    input_ids = tokenizer.apply_chat_template(
        chat, 
        tokenize=True, 
        return_tensors='pt', 
        continue_final_message=True
    )
    input_ids = input_ids.to('cuda')
    speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
    outputs = model.generate(
        input_ids,
        max_length=2048,
        eos_token_id=speech_end_id,
        do_sample=True,    
        top_p=1,
        temperature=0.8,
    )
    generated_ids = outputs[0][input_ids.shape[1]:-1]
    speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    speech_tokens = extract_speech_ids(speech_tokens)
    speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
    gen_wav = Codec_model.decode_code(speech_tokens)
sf.write("gen.wav", gen_wav[0, 0, :].cpu().numpy(), 16000)
