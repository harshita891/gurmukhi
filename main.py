# import torch
# import asyncio
# import logging
# import threading
# from tqdm import tqdm
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from IndicTransToolkit.processor import IndicProcessor
# from concurrent.futures import ThreadPoolExecutor
# import time

# # ---------------------- Logging Setup ----------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
#     datefmt="%H:%M:%S"
# )
# log = logging.getLogger()

# # --------------------- Device Setup -----------------------
# DEVICE = "cpu"
# log.info(f"Using device: {DEVICE}")

# # ------------------ Model Setup ---------------------------
# src_lang = "pan_Guru"

# model_name_en = "ai4bharat/indictrans2-indic-en-dist-200M"
# model_name_hi = "ai4bharat/indictrans2-indic-indic-dist-320M"

# tokenizer_en = AutoTokenizer.from_pretrained(model_name_en, trust_remote_code=True)
# model_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_en, trust_remote_code=True).to(DEVICE)

# tokenizer_hi = AutoTokenizer.from_pretrained(model_name_hi, trust_remote_code=True)
# model_hi = AutoModelForSeq2SeqLM.from_pretrained(model_name_hi, trust_remote_code=True).to(DEVICE)

# ip = IndicProcessor(inference=True)

# input_sentences = [
#     "ਜਦੋਂ ਮੈਂ ਛੋਟਾ ਸੀ, ਮੈਂ ਹਰ ਰੋਜ਼ ਪਾਰਕ ਜਾਂਦਾ ਸੀ।",
#     "ਅਸੀਂ ਪਿਛਲੇ ਹਫ਼ਤੇ ਇੱਕ ਨਵੀਂ ਫਿਲਮ ਵੇਖੀ ਜੋ ਬਹੁਤ ਪ੍ਰੇਰਣਾਦਾਇਕ ਸੀ।",
#     "ਜੇਕਰ ਤੁਸੀਂ ਮੈਨੂੰ ਉਸ ਸਮੇਂ ਮਿਲਦੇ, ਤਾਂ ਅਸੀਂ ਬਾਹਰ ਖਾਣਾ ਖਾਣੇ ਜਾਂਦੇ।",
#     "ਮੇਰੇ ਦੋਸਤ ਨੇ ਮੈਨੂੰ ਉਸਦੀ ਜਨਮਦਿਨ ਦੀ ਪਾਰਟੀ ਵਿੱਚ ਬੁਲਾਇਆ ਹੈ, ਅਤੇ ਮੈਂ ਉਸਨੂੰ ਇੱਕ ਤੋਹਫਾ ਦੇਵਾਂਗਾ।",
# ]

# translations_en = []
# translations_hi = []

# # ---------------- Translation Functions ---------------------

# def translate_to_english_sync():
#     log.info("Started Punjabi → English translation")
#     time.sleep(1)  # Simulate slight delay
#     translated = []
#     for sentence in tqdm(input_sentences, desc="Punjabi → English", position=0, leave=True):
#         batch_en = ip.preprocess_batch([sentence], src_lang=src_lang, tgt_lang="eng_Latn")
#         inputs_en = tokenizer_en(batch_en, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
#         with torch.no_grad():
#             outputs = model_en.generate(
#                 **inputs_en,
#                 max_length=256,
#                 num_beams=5,
#                 num_return_sequences=1,
#             )
#         with tokenizer_en.as_target_tokenizer():
#             decoded = tokenizer_en.batch_decode(outputs, skip_special_tokens=True)
#         result = ip.postprocess_batch(decoded, lang="eng_Latn")[0]
#         translated.append(result)
#         log.info(f"Translated: {sentence[:20]}... → {result[:20]}...")
#     return translated

# def translate_to_hindi_sync():
#     log.info("Started Punjabi → Hindi translation")
#     time.sleep(1)
#     translated = []
#     for sentence in tqdm(input_sentences, desc="Punjabi → Hindi", position=1, leave=True):
#         batch_hi = ip.preprocess_batch([sentence], src_lang=src_lang, tgt_lang="hin_Deva")
#         inputs_hi = tokenizer_hi(batch_hi, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
#         with torch.no_grad():
#             outputs = model_hi.generate(
#                 **inputs_hi,
#                 max_length=256,
#                 num_beams=5,
#                 num_return_sequences=1,
#             )
#         with tokenizer_hi.as_target_tokenizer():
#             decoded = tokenizer_hi.batch_decode(outputs, skip_special_tokens=True)
#         result = ip.postprocess_batch(decoded, lang="hin_Deva")[0]
#         translated.append(result)
#         log.info(f"Translated: {sentence[:20]}... → {result[:20]}...")
#     return translated

# # ---------------- Async Executor Wrapper ---------------------

# async def translate_async(executor):
#     loop = asyncio.get_event_loop()
#     log.info("Submitting translation tasks to threads")

#     task_en = loop.run_in_executor(executor, translate_to_english_sync)
#     task_hi = loop.run_in_executor(executor, translate_to_hindi_sync)

#     result_en, result_hi = await asyncio.gather(task_en, task_hi)

#     translations_en.extend(result_en)
#     translations_hi.extend(result_hi)

# # ---------------- Main Entry ---------------------

# async def main():
#     with ThreadPoolExecutor(max_workers=2) as executor:
#         await translate_async(executor)

# if __name__ == "__main__":
#     asyncio.run(main())

# # ---------------- Output Results ---------------------

# output_lines = []
# for input_sentence, en, hi in zip(input_sentences, translations_en, translations_hi):
#     print(f"\nPunjabi: {input_sentence}")
#     print(f"English: {en}")
#     print(f"Hindi: {hi}")
#     print("-" * 60)
#     output_lines.extend([
#         f"Punjabi: {input_sentence}\n",
#         f"English: {en}\n",
#         f"Hindi: {hi}\n",
#         "-" * 60 + "\n"
#     ])

# # ---------------- Save to File ---------------------

# with open("translations_output.txt", "w", encoding="utf-8") as f:
#     f.writelines(output_lines)

# log.info(" Translations saved to 'translations_output.txt'")

##### async only
# import torch
# import asyncio
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from IndicTransToolkit.processor import IndicProcessor


# DEVICE = "cpu"
# print(f"[INFO] Using device: {DEVICE}")


# src_lang = "pan_Guru"
# model_name_en = "ai4bharat/indictrans2-indic-en-dist-200M"
# model_name_hi = "ai4bharat/indictrans2-indic-indic-dist-320M"

# # Load tokenizers and models
# tokenizer_en = AutoTokenizer.from_pretrained(model_name_en, trust_remote_code=True)
# model_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_en, trust_remote_code=True).to(DEVICE)

# tokenizer_hi = AutoTokenizer.from_pretrained(model_name_hi, trust_remote_code=True)
# model_hi = AutoModelForSeq2SeqLM.from_pretrained(model_name_hi, trust_remote_code=True).to(DEVICE)

# # Preprocessor
# ip = IndicProcessor(inference=True)

# # Input sentences
# input_sentences = [
#     "ਜਦੋਂ ਮੈਂ ਛੋਟਾ ਸੀ, ਮੈਂ ਹਰ ਰੋਜ਼ ਪਾਰਕ ਜਾਂਦਾ ਸੀ।",
#     "ਅਸੀਂ ਪਿਛਲੇ ਹਫ਼ਤੇ ਇੱਕ ਨਵੀਂ ਫਿਲਮ ਵੇਖੀ ਜੋ ਬਹੁਤ ਪ੍ਰੇਰਣਾਦਾਇਕ ਸੀ।",
#     "ਜੇਕਰ ਤੁਸੀਂ ਮੈਨੂੰ ਉਸ ਸਮੇਂ ਮਿਲਦੇ, ਤਾਂ ਅਸੀਂ ਬਾਹਰ ਖਾਣਾ ਖਾਣੇ ਜਾਂਦੇ।",
#     "ਮੇਰੇ ਦੋਸਤ ਨੇ ਮੈਨੂੰ ਉਸਦੀ ਜਨਮਦਿਨ ਦੀ ਪਾਰਟੀ ਵਿੱਚ ਬੁਲਾਇਆ ਹੈ, ਅਤੇ ਮੈਂ ਉਸਨੂੰ ਇੱਕ ਤੋਹਫਾ ਦੇਵਾਂਗਾ।",
# ]

# translations_en = []
# translations_hi = []

# # Translation functions with async
# async def translate_to_english():
#     print("\n[INFO] Translating Punjabi → English")
#     batch_en = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang="eng_Latn")
#     inputs_en = tokenizer_en(batch_en, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

#     with torch.no_grad():
#         outputs = model_en.generate(
#             **inputs_en,
#             max_length=256,
#             num_beams=5,
#             num_return_sequences=1,
#         )

#     with tokenizer_en.as_target_tokenizer():
#         decoded = tokenizer_en.batch_decode(outputs, skip_special_tokens=True)
#     translations_en.extend(ip.postprocess_batch(decoded, lang="eng_Latn"))

# async def translate_to_hindi():
#     print("\n[INFO] Translating Punjabi → Hindi")
#     batch_hi = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang="hin_Deva")
#     inputs_hi = tokenizer_hi(batch_hi, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

#     with torch.no_grad():
#         outputs = model_hi.generate(
#             **inputs_hi,
#             max_length=256,
#             num_beams=5,
#             num_return_sequences=1,
#         )

#     with tokenizer_hi.as_target_tokenizer():
#         decoded = tokenizer_hi.batch_decode(outputs, skip_special_tokens=True)
#     translations_hi.extend(ip.postprocess_batch(decoded, lang="hin_Deva"))

# async def main():
    
#     task_en = asyncio.create_task(translate_to_english())
#     task_hi = asyncio.create_task(translate_to_hindi())

    
#     await asyncio.gather(task_en, task_hi)


# if __name__ == "__main__":
#     asyncio.run(main())


# output_lines = []
# for input_sentence, en, hi in zip(input_sentences, translations_en, translations_hi):
#     print(f"\nPunjabi: {input_sentence}")
#     print(f"English: {en}")
#     print(f"Hindi: {hi}")
#     print("-" * 60)
#     output_lines.extend([
#         f"Punjabi: {input_sentence}\n",
#         f"English: {en}\n",
#         f"Hindi: {hi}\n",
#         "-" * 60 + "\n"
#     ])

import torch
import asyncio
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

DEVICE = "cpu"
print(f"[INFO] Using device: {DEVICE}")


src_lang = "pan_Guru"
model_name_en = "ai4bharat/indictrans2-indic-en-dist-200M"
model_name_hi = "ai4bharat/indictrans2-indic-indic-dist-320M"


tokenizer_en = AutoTokenizer.from_pretrained(model_name_en, trust_remote_code=True)
model_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_en, trust_remote_code=True).to(DEVICE)

tokenizer_hi = AutoTokenizer.from_pretrained(model_name_hi, trust_remote_code=True)
model_hi = AutoModelForSeq2SeqLM.from_pretrained(model_name_hi, trust_remote_code=True).to(DEVICE)

ip = IndicProcessor(inference=True)

input_sentences = [
    "ਜਦੋਂ ਮੈਂ ਛੋਟਾ ਸੀ, ਮੈਂ ਹਰ ਰੋਜ਼ ਪਾਰਕ ਜਾਂਦਾ ਸੀ।",
    "ਅਸੀਂ ਪਿਛਲੇ ਹਫ਼ਤੇ ਇੱਕ ਨਵੀਂ ਫਿਲਮ ਵੇਖੀ ਜੋ ਬਹੁਤ ਪ੍ਰੇਰਣਾਦਾਇਕ ਸੀ।",
    "ਜੇਕਰ ਤੁਸੀਂ ਮੈਨੂੰ ਉਸ ਸਮੇਂ ਮਿਲਦੇ, ਤਾਂ ਅਸੀਂ ਬਾਹਰ ਖਾਣਾ ਖਾਣੇ ਜਾਂਦੇ।",
    "ਮੇਰੇ ਦੋਸਤ ਨੇ ਮੈਨੂੰ ਉਸਦੀ ਜਨਮਦਿਨ ਦੀ ਪਾਰਟੀ ਵਿੱਚ ਬੁਲਾਇਆ ਹੈ, ਅਤੇ ਮੈਂ ਉਸਨੂੰ ਇੱਕ ਤੋਹਫਾ ਦੇਵਾਂਗਾ।",
]

async def translate_batch(batch, src, tgt, tokenizer, model):
    preprocessed = ip.preprocess_batch(batch, src_lang=src, tgt_lang=tgt)
    inputs = tokenizer(preprocessed, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )
    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return ip.postprocess_batch(decoded, lang=tgt)

async def main():
    task_en = translate_batch(input_sentences, src_lang, "eng_Latn", tokenizer_en, model_en)
    task_hi = translate_batch(input_sentences, src_lang, "hin_Deva", tokenizer_hi, model_hi)

    translations_en, translations_hi = await asyncio.gather(task_en, task_hi)

    output_lines = []
    for punjabi, en, hi in zip(input_sentences, translations_en, translations_hi):
        print(f"\nPunjabi: {punjabi}")
        print(f"English: {en}")
        print(f"Hindi: {hi}")
        print("-" * 60)
        output_lines.extend([
            f"Punjabi: {punjabi}\n",
            f"English: {en}\n",
            f"Hindi: {hi}\n",
            "-" * 60 + "\n"
        ])

    with open("translations_output.txt", "w", encoding="utf-8") as f:
        f.writelines(output_lines)

    print("\nTrans saved.")

if __name__ == "__main__":
    asyncio.run(main())
