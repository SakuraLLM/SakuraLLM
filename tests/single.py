import sys
import requests
import argparse
import logging
from pprint import pprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('url', type=str, nargs='*', default='http://127.0.0.1:5000')
parser.add_argument('--auth', type=str, nargs='*', default='sakura:itsmygo')
parser.add_argument('--prefix', type=int, nargs='*', default=None)

args = parser.parse_args()
logger.info(args)

session = requests.Session()
auth = args.auth.split(':')
session.auth = (auth[0], auth[1])

endpoint = f"{args.url}/api/v1/generate"

prompt = f"<reserved_106>将下面的日文文本翻译成中文: {args.prefix} アスカム子爵家長女、アデル・フォン・アスカムは、１０歳になったある日、強烈な頭痛と共に全てを思い出した。 　自分が以前、栗原海里（くりはらみさと）という名の１８歳の日本人であったこと、幼い少女を助けようとして命を落としたこと、そして、神様に出会ったことを……。 　少々出来が良過ぎたために周りの期待が大きく、思うように生きることができなかった海里は、望みを尋ねる神様にこうお願いしたのであった。 『次の人生、能力は平均値でお願いします！』 　なのに、何だか話が違うよ！ 　３つの名前を持つ少女、うっかりＳ級ハンターなんかにならないように気を付けて、普通に生きて行きます。 　だって、私はごく普通の、平凡な女の子なんだからね。いや、ホント。 １～１３巻まで、アース・スターノベルから書籍化。 １４巻（２０２１年１月７日刊行）からは、スクエニの新レーベル、『ＳＱＥＸノベル』から刊行。 コミックス（アース・スターノベル）、スピンオフコミックス『私、日常は平均値でって言ったよね！』（アース・スターノベル）、リブートコミックス（スクウェア・エニックス）共々、よろしくお願いいたします。(^^)/ <reserved_107>"
request = {
    "prompt": prompt,
    "auto_max_new_tokens": False,
    "max_tokens_second": 0,
    # Generation params. If 'preset' is set to different than 'None', the values
    # in presets/preset-name.yaml are used instead of the individual numbers.
    "preset": "None",
    "max_new_tokens": 1024,
    "do_sample": True,
    "temperature": 0.1,
    "top_p": 0.3,
    "repetition_penalty": 1.0,
    "num_beams": 1,
    "typical_p": 1,
    "epsilon_cutoff": 0,  # In units of 1e-4
    "eta_cutoff": 0,  # In units of 1e-4
    "tfs": 1,
    "top_a": 0,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "repetition_penalty_range": 0,
    "top_k": 40,
    "min_length": 0,
    "no_repeat_ngram_size": 0,
    "penalty_alpha": 0,
    "length_penalty": 1,
    "early_stopping": False,
    "mirostat_mode": 0,
    "mirostat_tau": 5,
    "mirostat_eta": 0.1,
    "grammar_string": "",
    "guidance_scale": 1,
    "negative_prompt": "",
    "seed": -1,
    "add_bos_token": True,
    "truncation_length": 2048,
    "ban_eos_token": False,
    "custom_token_bans": "",
    "skip_special_tokens": True,
    "stopping_strings": [],
}
pprint(request)

response = session.post(endpoint, json=request)
result = response.json()
pprint(result)

result = result["results"][0]["text"]
pprint(result)
