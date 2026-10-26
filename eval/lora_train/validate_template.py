"""Valida o FIXED_TEMPLATE ({% generation %}) contra o tokenizer do AMALIA."""
from transformers import AutoTokenizer

FIXED = (
    "{% if messages[0]['role'] != 'system' %}"
    "{% set messages = [{'role': 'system', 'content': 'O teu nome é AMALIA, e és um modelo avançado de linguagem útil. Responde sempre na língua do utilizador, a menos que sejas instruído em contrário, e lembra-te que a tua língua principal é o português europeu.'}] + messages %}"
    "{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\n' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{ '<|im_end|>\n' }}"
    "{% else %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

tok = AutoTokenizer.from_pretrained("amalia-llm/AMALIA-9B-0626-SFT")
tok.chat_template = FIXED

# 1) render identico ao template original?
orig = AutoTokenizer.from_pretrained("amalia-llm/AMALIA-9B-0626-SFT")
msgs = [{"role": "user", "content": "Quando nasceu?"},
        {"role": "assistant", "content": "Nasceu em 1850."},
        {"role": "user", "content": "E morreu?"}]
t_new = tok.apply_chat_template(msgs, tokenize=False)
t_orig = orig.apply_chat_template(msgs, tokenize=False)
print("render igual ao original:", t_new == t_orig)
if t_new != t_orig:
    print("NOVO:", repr(t_new[:300]))
    print("ORIG:", repr(t_orig[:300]))

# 2) mascara assistant-only
enc = tok.apply_chat_template(msgs, tokenize=True,
                              return_assistant_tokens_mask=True, return_dict=True)
ids, mask = enc["input_ids"], enc["assistant_masks"]
print("tokens:", len(ids), "| assistant tokens:", sum(mask))
print("zona mascarada:", repr(tok.decode([i for i, m in zip(ids, mask) if m])))
assert sum(mask) > 0 and sum(mask) < len(ids)
print("TEMPLATE OK - assistant-only loss funcionara")
