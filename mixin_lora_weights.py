from peft import PeftModel
from transformers import OPTForCausalLM, AutoTokenizer

BASE_MODEL = "facebook/opt-6.7b"

base_model = OPTForCausalLM.from_pretrained(
    BASE_MODEL,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

lora_model = PeftModel.from_pretrained(base_model, "alpaca-opt-6.7b")

lora_model.train(False)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

OPTForCausalLM.save_pretrained(
    base_model, "./hf_ckpt", state_dict=deloreanized_sd, max_shard_size="400MB"
)

# now load as usual
mixin_model = OPTForCausalLM.from_pretrained(
    "./hf_ckpt",
).push_to_hub("alpaca-opt-6.7b")
tokenizer.push_to_hub("alpaca-opt-6.7b")
