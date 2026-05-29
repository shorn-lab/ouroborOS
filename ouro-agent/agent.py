import sys, json, re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

MODEL_PATH = "/opt/ouro-model"

# Load model once
config = AutoConfig.from_pretrained(MODEL_PATH)
config.total_ut_steps = 4  # Adjustable: fewer = faster, less reasoning depth

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config,
    device_map="auto",
    torch_dtype=torch.float16  # Use float16 to save RAM
)
model.eval()

SYSTEM_PROMPT = """You are a helpful AI assistant running locally on an external SSD.
You think step by step. You are an agent that can reason through problems carefully.
When you need to use a tool, output: TOOL: <tool_name>(<args>)
Available tools: shell(cmd), read_file(path), write_file(path, content)
"""

def run_tool(tool_call: str) -> str:
    """Execute a tool call and return result."""
    import subprocess
    match = re.match(r'(\w+)\((.+)\)', tool_call, re.DOTALL)
    if not match:
        return "Error: malformed tool call"
    name, args = match.group(1), match.group(2).strip('"\'')
    if name == "shell":
        result = subprocess.run(args, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout + result.stderr
    elif name == "read_file":
        try:
            return open(args).read()
        except Exception as e:
            return str(e)
    elif name == "write_file":
        parts = args.split(",", 1)
        if len(parts) == 2:
            open(parts[0].strip().strip('"'), 'w').write(parts[1].strip().strip('"'))
            return "Written."
    return "Unknown tool"

def chat(history: list[dict], user_input: str) -> str:
    history.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    # Handle tool calls
    if "TOOL:" in response:
        tool_line = re.search(r'TOOL:\s*(.+)', response)
        if tool_line:
            tool_result = run_tool(tool_line.group(1).strip())
            response += f"\n[Tool result: {tool_result}]"
    history.append({"role": "assistant", "content": response})
    return response
