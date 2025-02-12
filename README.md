# IndicTransToolkit

## About
The goal of this repository is to provide a simple, modular, and extendable toolkit for [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) and be compatible with the HuggingFace models released. Please refer to the `CHANGELOG.md` for latest developments.

## Pre-requisites
 - `Python 3.8+`
 - [Indic NLP Library](https://github.com/VarunGumma/indic_nlp_library)
 - Other requirements as listed in `requirements.txt`

## Configuration
 - Editable installation (Note, this may take a while):
```bash 
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit

pip install --editable . --use-pep517 # required for pip >= 25.0

# in case it fails, try:
# pip install --editable . --use-pep517 --config-settings editable_mode=compat
```

## Examples
For the training usecase, please refer [here](https://github.com/AI4Bharat/IndicTrans2/tree/main/huggingface_interface).

### PreTainedTokenizer 
```python
import torch
from IndicTransToolkit import IndicProcessor # NOW IMPLEMENTED IN CYTHON !!
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ip = IndicProcessor(inference=True)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)

sentences = [
    "This is a test sentence.",
    "This is another longer different test sentence.",
    "Please send an SMS to 9876543210 and an email on newemail123@xyz.com by 15th October, 2023.",
]

batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva", visualize=False) # set it to visualize=True to print a progress bar
batch = tokenizer(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt")

with torch.inference_mode():
    outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)

with tokenizer.as_target_tokenizer():
    # This scoping is absolutely necessary, as it will instruct the tokenizer to tokenize using the target vocabulary.
    # Failure to use this scoping will result in gibberish/unexpected predictions as the output will be de-tokenized with the source vocabulary instead.
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

outputs = ip.postprocess_batch(outputs, lang="hin_Deva")
print(outputs)

>>> ['यह एक परीक्षण वाक्य है।', 'यह एक और लंबा अलग परीक्षण वाक्य है।', 'कृपया 9876543210 पर एक एस. एम. एस. भेजें और 15 अक्टूबर, 2023 तक newemail123@xyz.com पर एक ईमेल भेजें।']
```

### Evaluation
- `IndicEvaluator` is a python implementation of [compute_metrics.sh](https://github.com/AI4Bharat/IndicTrans2/blob/main/compute_metrics.sh). 
- We have found that this python implementation gives slightly lower scores than the original `compute_metrics.sh`. So, please use this function cautiously, and feel free to raise a PR if you have found the bug/fix. 
```python
from IndicTransToolkit import IndicEvaluator

# this method returns a dictionary with BLEU and ChrF2++ scores with appropriate signatures
evaluator = IndicEvaluator()
scores = evaluator.evaluate(tgt_lang=tgt_lang, preds=pred_file, refs=ref_file) 

# alternatively, you can pass the list of predictions and references instead of files 
# scores = evaluator.evaluate(tgt_lang=tgt_lang, preds=preds, refs=refs)
```

## Authors
 - Varun Gumma (varun230999@gmail.com)
 - Jay Gala (jaygala24@gmail.com)
 - Pranjal Agadh Chitale (pranjalchitale@gmail.com)
 - Raj Dabre (prajdabre@gmail.com)


## Bugs and Contribution
Since this a bleeding-edge module, you may encounter broken stuff and import issues once in a while. In case you encounter any bugs or want additional functionalities, please feel free to raise `Issues`/`Pull Requests` or contact the authors. 


## Citation
If you use our codebase, or models, please do cite the following paper:
```bibtex
@article{
    gala2023indictrans,
    title={IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
    author={Jay Gala and Pranjal A Chitale and A K Raghavan and Varun Gumma and Sumanth Doddapaneni and Aswanth Kumar M and Janki Atul Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M Khapra and Raj Dabre and Anoop Kunchukuttan},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2023},
    url={https://openreview.net/forum?id=vfT4YuzAYA},
    note={}
}
```