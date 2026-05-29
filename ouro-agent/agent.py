import sys
import types
import warnings
import sys
import types
import warnings

def _patch_transformers():
    """Patch all missing functions for Ouro model compatibility."""
    
    # 1. Patch layer_type_validation
    try:
        from transformers.configuration_utils import layer_type_validation
        print("✓ layer_type_validation already available")
    except ImportError:
        print("⚠️ Patching missing layer_type_validation")
        def _layer_type_validation(layer_types, allowed_types=None):
            """Mock validation that always passes."""
            if allowed_types is None:
                allowed_types = ['full_attention', 'sliding_attention']
            for lt in layer_types:
                if lt not in allowed_types:
                    warnings.warn(f"Unknown layer type: {lt}")
            return True
        
        import transformers.configuration_utils as cu
        cu.layer_type_validation = _layer_type_validation
    
    # 2. Patch rope_config_validation
    try:
        from transformers.modeling_rope_utils import rope_config_validation
        print("✓ rope_config_validation already available")
    except ImportError:
        print("⚠️ Patching missing rope_config_validation")
        def _rope_config_validation(config):
            """Mock rope validation that passes."""
            return True
        
        import transformers.modeling_rope_utils as ru
        ru.rope_config_validation = _rope_config_validation
    
    # 3. Patch compute_default_rope_parameters (the one you're missing)
    try:
        from transformers.modeling_rope_utils import _compute_default_rope_parameters
        print("✓ _compute_default_rope_parameters already available")
    except ImportError:
        print("⚠️ Patching missing _compute_default_rope_parameters")
        
        def _compute_default_rope_parameters(config, device=None, seq_len=None, **rope_kwargs):
            """
            Compute default RoPE parameters for Ouro model.
            Based on implementation from transformers 4.54.1
            """
            import torch
            
            # Handle different input patterns
            if config is not None:
                base = getattr(config, "rope_theta", 10000.0)
                partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
                head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
                dim = int(head_dim * partial_rotary_factor)
            elif len(rope_kwargs) > 0:
                base = rope_kwargs.get("base", 10000.0)
                dim = rope_kwargs.get("dim", 128)
            else:
                base = 10000.0
                dim = 128
            
            # Compute inverse frequencies
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
            attention_factor = 1.0
            
            return inv_freq, attention_factor
        
        # Patch both the public and private versions
        import transformers.modeling_rope_utils as ru
        ru._compute_default_rope_parameters = _compute_default_rope_parameters
        ru.compute_default_rope_parameters = _compute_default_rope_parameters  # Some versions may expect this name
    
    # 4. Also patch compute_llama3_parameters if needed (for some model variants)
    try:
        from transformers.modeling_rope_utils import _compute_llama3_parameters
        print("✓ _compute_llama3_parameters already available")
    except ImportError:
        print("⚠️ Patching missing _compute_llama3_parameters")
        
        def _compute_llama3_parameters(config, device, seq_len, **rope_kwargs):
            """Fallback for llama3-style RoPE parameters"""
            return _compute_default_rope_parameters(config, device, seq_len, **rope_kwargs)
        
        import transformers.modeling_rope_utils as ru
        ru._compute_llama3_parameters = _compute_llama3_parameters
    
    # 5. Ensure ALLOWED_LAYER_TYPES exists
    try:
        import transformers.configuration_utils as cu
        if not hasattr(cu, 'ALLOWED_LAYER_TYPES'):
            cu.ALLOWED_LAYER_TYPES = [
                'full_attention', 'sliding_attention', 'full_attention_no_rope',
                'sliding_attention_no_rope', 'full_attention_with_rope', 
                'sliding_attention_with_rope', 'dummy', 'sub_attention'
            ]
            print("✓ ALLOWED_LAYER_TYPES added")
    except:
        pass
    
    # 6. Add rope_validation_forward function if missing
    try:
        from transformers.modeling_rope_utils import rope_validation_forward
    except ImportError:
        def rope_validation_forward(config, **kwargs):
            """Forward validation for RoPE parameters"""
            return True
        
        import transformers.modeling_rope_utils as ru
        ru.rope_validation_forward = rope_validation_forward
    
    print("✓ All transformers patches applied successfully")

# Apply patches IMMEDIATELY
_patch_transformers()

# Now import the rest of your dependencies
import os
import re
import subprocess
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

# Continue with your normal code...
MODEL_PATH = str(Path.home() / "ouro-model")

# Optional: Add version check for debugging
print(f"📦 Transformers version: {transformers.__version__}")
print(f"📦 PyTorch version: {torch.__version__}")

# Load your model as usual
config = AutoConfig.from_pretrained(MODEL_PATH)
config.total_ut_steps = 4

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()


# Enhanced system prompt for full system access
SYSTEM_PROMPT = """You are a powerful AI assistant with FULL SYSTEM ACCESS running locally on an external SSD.
You have the ability to execute ANY command, install software, modify system files, and perform any administrative task.
Think step by step before executing commands that could affect system stability.

TOOL FORMATS:
- TOOL: shell(<command>) - Execute ANY shell command (use sudo when needed)
- TOOL: shell_background(<command>) - Run command in background
- TOOL: read_file(<path>) - Read any file
- TOOL: write_file(<path>, <content>) - Write to any file
- TOOL: append_file(<path>, <content>) - Append to file
- TOOL: delete_file(<path>) - Delete file/directory
- TOOL: move_file(<source>, <dest>) - Move/rename files
- TOOL: copy_file(<source>, <dest>) - Copy files
- TOOL: change_permissions(<path>, <mode>) - chmod
- TOOL: change_owner(<path>, <owner>) - chown
- TOOL: list_directory(<path>) - List directory contents
- TOOL: get_system_info() - Get system information
- TOOL: install_package(<package>) - Install via xbps (Void Linux)
- TOOL: search_packages(<query>) - Search for packages
- TOOL: check_service(<service>) - Check service status
- TOOL: start_service(<service>) - Start a service
- TOOL: stop_service(<service>) - Stop a service
- TOOL: reboot_system() - Reboot the system
- TOOL: shutdown_system() - Shutdown the system

IMPORTANT: 
- Always ask for confirmation before destructive operations (rm -rf, formatting, etc.)
- Use sudo prefix when commands require root privileges
- Think step by step about potential consequences
- Cache results of expensive operations when possible

Be helpful, efficient, and safety-conscious despite having full access."""

class SystemTools:
    """Handles system-level operations with full access"""
    
    def __init__(self):
        self.command_history = []
        self.background_processes = []
        
    def run_shell(self, command: str, timeout: int = 60, background: bool = False) -> str:
        """Execute shell command with full privileges"""
        # Log for auditing
        timestamp = datetime.now().isoformat()
        self.command_history.append(f"[{timestamp}] $ {command}")
        
        if background:
            # Run in background without waiting
            process = subprocess.Popen(
                command, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                executable='/bin/bash'
            )
            self.background_processes.append(process)
            return f"Background process started with PID: {process.pid}"
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                executable='/bin/bash'
            )
            
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[STDERR]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[Exit code: {result.returncode}]"
            
            return output.strip() or "Command executed successfully (no output)"
            
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def install_package(self, package: str) -> str:
        """Install package using xbps (Void Linux)"""
        return self.run_shell(f"sudo xbps-install -Sy {package}")
    
    def search_packages(self, query: str) -> str:
        """Search for packages"""
        return self.run_shell(f"xbps-query -Rs {query}")
    
    def get_system_info(self) -> str:
        """Gather system information"""
        info = []
        info.append(self.run_shell("uname -a"))
        info.append(self.run_shell("cat /etc/os-release | head -5"))
        info.append(self.run_shell("df -h / | tail -1"))
        info.append(self.run_shell("free -h"))
        info.append(self.run_shell("whoami"))
        return "\n".join(info)
    
    def read_file(self, path: str) -> str:
        """Read any file on the system"""
        expanded_path = Path(path).expanduser().resolve()
        try:
            if not expanded_path.exists():
                return f"Error: File not found: {path}"
            with open(expanded_path, 'r') as f:
                content = f.read()
                # Truncate very large files
                if len(content) > 50000:
                    content = content[:50000] + "\n... (truncated, file too large)"
                return content
        except PermissionError:
            # Try with sudo if permission denied
            return self.run_shell(f"sudo cat {expanded_path}")
        except Exception as e:
            return f"Error: {str(e)}"
    
    def write_file(self, path: str, content: str) -> str:
        """Write to any file (will use sudo if needed)"""
        expanded_path = Path(path).expanduser().resolve()
        try:
            # Ensure directory exists
            expanded_path.parent.mkdir(parents=True, exist_ok=True)
            with open(expanded_path, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except PermissionError:
            # Use sudo via tee
            return self.run_shell(f"echo {shlex.quote(content)} | sudo tee {expanded_path} > /dev/null")
        except Exception as e:
            return f"Error: {str(e)}"
    
    def append_file(self, path: str, content: str) -> str:
        """Append to any file"""
        expanded_path = Path(path).expanduser().resolve()
        try:
            with open(expanded_path, 'a') as f:
                f.write(content + "\n")
            return f"Successfully appended to {path}"
        except PermissionError:
            return self.run_shell(f"echo {shlex.quote(content)} | sudo tee -a {expanded_path} > /dev/null")
    
    def delete_file(self, path: str) -> str:
        """Delete file or directory (use with caution)"""
        expanded_path = Path(path).expanduser().resolve()
        if expanded_path.is_dir():
            return self.run_shell(f"rm -rf {expanded_path}")
        else:
            return self.run_shell(f"rm {expanded_path}")
    
    def move_file(self, source: str, dest: str) -> str:
        """Move/rename files"""
        return self.run_shell(f"mv {shlex.quote(source)} {shlex.quote(dest)}")
    
    def copy_file(self, source: str, dest: str) -> str:
        """Copy files"""
        return self.run_shell(f"cp {shlex.quote(source)} {shlex.quote(dest)}")
    
    def change_permissions(self, path: str, mode: str) -> str:
        """Change file permissions"""
        return self.run_shell(f"chmod {mode} {shlex.quote(path)}")
    
    def change_owner(self, path: str, owner: str) -> str:
        """Change file ownership"""
        return self.run_shell(f"sudo chown {owner} {shlex.quote(path)}")
    
    def list_directory(self, path: str = ".") -> str:
        """List directory contents with details"""
        return self.run_shell(f"ls -lah {shlex.quote(path)}")
    
    def check_service(self, service: str) -> str:
        """Check service status"""
        return self.run_shell(f"sv status {service}")
    
    def start_service(self, service: str) -> str:
        """Start a service"""
        return self.run_shell(f"sudo sv start {service}")
    
    def stop_service(self, service: str) -> str:
        """Stop a service"""
        return self.run_shell(f"sudo sv stop {service}")
    
    def reboot_system(self) -> str:
        """Reboot the system"""
        return self.run_shell("sudo reboot")
    
    def shutdown_system(self) -> str:
        """Shutdown the system"""
        return self.run_shell("sudo shutdown -h now")
    
    def get_command_history(self) -> str:
        """Return command history for auditing"""
        return "\n".join(self.command_history[-20:])  # Last 20 commands

# Initialize tools
tools = SystemTools()

def run_tool(tool_call: str) -> str:
    """Execute any tool with full system access"""
    # Parse tool name and arguments
    match = re.match(r'(\w+)\((.*)\)', tool_call, re.DOTALL)
    if not match:
        return "Error: malformed tool call"
    
    name = match.group(1)
    args_str = match.group(2).strip()
    
    # Route to appropriate tool function
    if name == "shell":
        # Remove surrounding quotes if present
        args = args_str.strip('"\'')
        return tools.run_shell(args)
    
    elif name == "shell_background":
        args = args_str.strip('"\'')
        return tools.run_shell(args, background=True)
    
    elif name == "read_file":
        filepath = args_str.strip('"\'')
        return tools.read_file(filepath)
    
    elif name == "write_file":
        # Parse path and content
        first_quote = args_str.find('"')
        if first_quote == -1:
            return "Error: Invalid format. Use: write_file('path', 'content')"
        path_end = args_str.find('"', first_quote + 1)
        if path_end == -1:
            return "Error: Invalid path format"
        filepath = args_str[first_quote + 1:path_end]
        content_start = args_str.find(',', path_end) + 1
        content_start = args_str.find('"', content_start)
        if content_start == -1:
            return "Error: Invalid content format"
        content_end = args_str.rfind('"')
        content = args_str[content_start + 1:content_end]
        return tools.write_file(filepath, content)
    
    elif name == "append_file":
        # Similar parsing to write_file
        first_quote = args_str.find('"')
        path_end = args_str.find('"', first_quote + 1)
        filepath = args_str[first_quote + 1:path_end]
        content_start = args_str.find(',', path_end) + 1
        content_start = args_str.find('"', content_start)
        content_end = args_str.rfind('"')
        content = args_str[content_start + 1:content_end]
        return tools.append_file(filepath, content)
    
    elif name == "delete_file":
        filepath = args_str.strip('"\'')
        # Add confirmation for dangerous paths
        dangerous_paths = ['/', '/home', '/etc', '/usr', '/var', '/boot']
        if any(filepath.startswith(dp) for dp in dangerous_paths):
            return f"⚠️ DESTRUCTIVE OPERATION BLOCKED: Deleting '{filepath}' would damage your system. Manual approval required."
        return tools.delete_file(filepath)
    
    elif name == "move_file":
        parts = args_str.split(',')
        if len(parts) != 2:
            return "Error: Need source and destination"
        source = parts[0].strip().strip('"\'')
        dest = parts[1].strip().strip('"\'')
        return tools.move_file(source, dest)
    
    elif name == "copy_file":
        parts = args_str.split(',')
        if len(parts) != 2:
            return "Error: Need source and destination"
        source = parts[0].strip().strip('"\'')
        dest = parts[1].strip().strip('"\'')
        return tools.copy_file(source, dest)
    
    elif name == "change_permissions":
        parts = args_str.split(',')
        if len(parts) != 2:
            return "Error: Need path and mode"
        filepath = parts[0].strip().strip('"\'')
        mode = parts[1].strip().strip('"\'')
        return tools.change_permissions(filepath, mode)
    
    elif name == "change_owner":
        parts = args_str.split(',')
        if len(parts) != 2:
            return "Error: Need path and owner"
        filepath = parts[0].strip().strip('"\'')
        owner = parts[1].strip().strip('"\'')
        return tools.change_owner(filepath, owner)
    
    elif name == "list_directory":
        path = args_str.strip('"\'') if args_str else "."
        return tools.list_directory(path)
    
    elif name == "get_system_info":
        return tools.get_system_info()
    
    elif name == "install_package":
        package = args_str.strip('"\'')
        return tools.install_package(package)
    
    elif name == "search_packages":
        query = args_str.strip('"\'')
        return tools.search_packages(query)
    
    elif name == "check_service":
        service = args_str.strip('"\'')
        return tools.check_service(service)
    
    elif name == "start_service":
        service = args_str.strip('"\'')
        return tools.start_service(service)
    
    elif name == "stop_service":
        service = args_str.strip('"\'')
        return tools.stop_service(service)
    
    elif name == "reboot_system":
        return tools.reboot_system()
    
    elif name == "shutdown_system":
        return tools.shutdown_system()
    
    elif name == "command_history":
        return tools.get_command_history()
    
    else:
        return f"Unknown tool: {name}. Available tools: shell, read_file, write_file, append_file, delete_file, move_file, copy_file, change_permissions, change_owner, list_directory, get_system_info, install_package, search_packages, check_service, start_service, stop_service, reboot_system, shutdown_system"

def chat(history: list[dict], user_input: str) -> str:
    """Process user input with full system access"""
    history.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback formatting
        text = f"System: {SYSTEM_PROMPT}\n"
        for msg in messages[1:]:
            text += f"{msg['role'].capitalize()}: {msg['content']}\n"
        text += "Assistant: "
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Handle multiple tool calls (one per line)
    lines = response.split('\n')
    modified_response = []
    tool_results = []
    
    for line in lines:
        tool_pattern = r'TOOL:\s*(\w+\([^)]*\))'
        tool_match = re.search(tool_pattern, line)
        if tool_match:
            tool_call = tool_match.group(1)
            result = run_tool(tool_call)
            tool_results.append(f"🔧 Executed: {tool_call}\n📋 Result: {result}")
            # Replace tool line with result in response
            modified_response.append(f"[Tool executed: {tool_call}]")
        else:
            modified_response.append(line)
    
    if tool_results:
        final_response = "\n".join(modified_response) + "\n\n" + "\n\n---\n\n".join(tool_results)
    else:
        final_response = "\n".join(modified_response)
    
    history.append({"role": "assistant", "content": final_response})
    return final_response

# Interactive main loop
if __name__ == "__main__":
    print("="*60)
    print("🤖 AI AGENT WITH FULL SYSTEM ACCESS")
    print("="*60)
    print("⚠️  WARNING: This agent can execute ANY command on your system")
    print("⚠️  The agent can install software, modify files, and change system settings")
    print("⚠️  Always review tool executions before allowing destructive operations")
    print("="*60)
    print("\nCommands:")
    print("  Type your requests naturally")
    print("  'exit' - Quit the agent")
    print("  'history' - Show command execution history")
    print("  'clear' - Clear conversation")
    print("="*60)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Shutting down agent...")
                break
            elif user_input.lower() == 'clear':
                conversation_history = []
                print("🧹 Conversation cleared")
                continue
            elif user_input.lower() == 'history':
                print("\n📜 Command History:")
                print(tools.get_command_history())
                continue
            
            if not user_input:
                continue
            
            print("🤔 Agent thinking...")
            response = chat(conversation_history, user_input)
            print(f"🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")