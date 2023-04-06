import torch
from peft import PeftModel
from transformers import AutoTokenizer, OPTForCausalLM

MODEL_NAME = "facebook/opt-6.7b"
# MODEL_NAME = "facebook/opt-125m"

if torch.cuda.is_available():
    model = OPTForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model, "alpaca-opt-6.7b", torch_dtype=torch.float16
    )
    device = "cuda"
else:
    model = OPTForCausalLM.from_pretrained(
        MODEL_NAME,
    )
    model = PeftModel.from_pretrained(model, "alpaca-opt-6.7b")
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def generate_prompt(instruction, input=None):
    base_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n" + instruction

    if input:
        input_section = "\n\n### Input:\n" + input
    else:
        input_section = ""

    response_section = "\n\n### Response:"

    return base_prompt + input_section + response_section


model.eval()


def evaluate(instruction, input=None):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_output = model.generate(
        input_ids=input_ids,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=True,
        max_length=256,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
    )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


if __name__ == "__main__":
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
