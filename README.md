Training and inference code for (OPT)[www.facebook.com] models based on (Alpaca)[www.stanford.com]. You can get the trained model
from the huggingface hub.

```python
from transfomers import Model, tokenizer
model = Model("manuel030/alpaca-opt-6.7b")
```

Why?
Base model of Standford alpaca doesn't have permissive license. 

Credits
alpaca-lora