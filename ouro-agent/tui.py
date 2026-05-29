from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog, Static, TabbedContent, TabPane
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual import work
from textual.binding import Binding
from datetime import datetime
import time
import agent  # Your agent module

# ── Colour palette ────────────────────────────────────────────────────────────
PURPLE   = "#5050d0"
PURPLE_D = "#2a2d60"
TEAL     = "#207050"
TEAL_D   = "#0f2a20"
MUTED    = "#3a4060"
GHOST    = "#252840"
BG_MAIN  = "#0b0c14"
BG_SIDE  = "#0d0f1c"
BG_INPUT = "#10121e"
TEXT     = "#b0b8d8"
TEXT_DIM = "#505870"

class OuroApp(App):
    """Ouro-2.6B-Thinking — portable AI agent TUI."""

    CSS = f"""
    Screen {{
        background: {BG_MAIN};
        color: {TEXT};
    }}

    /* ── Layout ── */
    #root-layout {{
        layout: horizontal;
        height: 100%;
    }}

    /* ── Sidebar ── */
    #sidebar {{
        width: 24;
        background: {BG_SIDE};
        border-right: solid {GHOST};
        height: 100%;
        padding: 0;
    }}
    #sidebar-title {{
        background: #10121e;
        border-bottom: solid {GHOST};
        padding: 1 2;
        color: {PURPLE};
        text-style: bold;
        width: 100%;
        height: 3;
    }}
    .nav-item {{
        padding: 1 2;
        color: {MUTED};
        height: 3;
    }}
    .nav-item:hover {{
        color: #8a90c0;
        background: #13162a;
    }}
    .nav-item.active {{
        color: {PURPLE};
        border-left: solid {PURPLE};
        background: #13162a;
    }}
    #model-info {{
        dock: bottom;
        border-top: solid {GHOST};
        padding: 1 2;
        color: {MUTED};
        background: {BG_SIDE};
        height: 6;
    }}
    #model-name {{
        color: {PURPLE};
        text-style: bold;
    }}

    /* ── Main area ── */
    #main-area {{
        layout: vertical;
        height: 100%;
        width: 1fr;
    }}

    /* ── Tabs ── */
    TabbedContent {{
        height: 1fr;
    }}
    TabPane {{
        padding: 0;
    }}

    /* ── Chat log ── */
    #chat-log {{
        height: 1fr;
        border: none;
        background: {BG_MAIN};
        padding: 1 2;
        scrollbar-color: {GHOST};
    }}
    #chat-log:focus {{
        scrollbar-color: {MUTED};
    }}

    /* ── Input bar ── */
    #input-area {{
        height: 5;
        border-top: solid {GHOST};
        background: {BG_INPUT};
        padding: 1 2;
    }}
    #user-input {{
        background: {BG_MAIN};
        border: solid {GHOST};
        color: {TEXT};
        width: 100%;
        height: 3;
    }}
    #user-input:focus {{
        border: solid {PURPLE_D};
    }}

    /* ── Status bar ── */
    #status-bar {{
        dock: bottom;
        height: 3;
        background: #090b18;
        border-top: solid {GHOST};
        color: {MUTED};
        padding: 0 2;
        layout: horizontal;
        align: center middle;
    }}
    .status-item {{
        margin-right: 2;
        color: {MUTED};
        height: 3;
        padding: 0 1;
        content-align: center middle;
    }}
    #loop-badge {{
        dock: right;
        color: {TEAL};
        height: 3;
        padding: 0 1;
        content-align: center middle;
    }}

    /* ── Message styles ── */
    .msg-you    {{ color: #8890c0; margin-bottom: 1; }}
    .msg-ouro   {{ color: {TEXT}; margin-bottom: 1; }}
    .msg-think  {{ color: {TEXT_DIM}; margin-bottom: 0; }}
    .msg-system {{ color: {TEAL}; margin-bottom: 1; }}
    .msg-tool   {{ color: #3a5040; margin-bottom: 1; }}

    Header {{
        display: none;
    }}
    Footer {{
        background: #090b18;
        color: {MUTED};
        height: 3;
    }}
    """

    BINDINGS = [
        Binding("ctrl+q", "quit",          "Quit",    show=True),
        Binding("ctrl+t", "focus_tools",   "Tools",   show=True),
        Binding("ctrl+m", "focus_memory",  "Memory",  show=True),
        Binding("ctrl+l", "clear_chat",    "Clear",   show=True),
        Binding("ctrl+s", "save_session",  "Save",    show=False),
    ]

    history = reactive([])
    uptime_seconds = reactive(0)
    turn_count = reactive(0)

    def __init__(self):
        super().__init__()
        self._start_time = time.time()

    def compose(self) -> ComposeResult:
        with Horizontal(id="root-layout"):
            # ── Sidebar ──────────────────────────────────────────────
            with Vertical(id="sidebar"):
                yield Static("⟳ OURO AGENT", id="sidebar-title")
                yield Static("› chat",   classes="nav-item active", id="nav-chat")
                yield Static("  tools",  classes="nav-item", id="nav-tools")
                yield Static("  memory", classes="nav-item", id="nav-memory")
                yield Static("  config", classes="nav-item", id="nav-config")
                with Static(id="model-info"):
                    yield Static("Ouro-2.6B-Thinking", id="model-name")
                    yield Static("steps: 4  float16", classes="status-item")
                    yield Static("ctx: 8192 tokens",  classes="status-item")

            # ── Main ─────────────────────────────────────────────────
            with Vertical(id="main-area"):
                with TabbedContent(initial="chat") as tabs:
                    with TabPane("Chat", id="chat-pane"):
                        with Vertical():
                            yield RichLog(
                                id="chat-log",
                                markup=True,
                                wrap=True,
                                highlight=True,
                            )
                            with Vertical(id="input-area"):
                                yield Input(
                                    placeholder="› ask anything...",
                                    id="user-input",
                                )
                    with TabPane("Tools", id="tools-pane"):
                        yield RichLog(id="tools-log", markup=True, wrap=True)
                    with TabPane("Memory", id="memory-pane"):
                        yield RichLog(id="memory-log", markup=True, wrap=True)
                    with TabPane("Config", id="config-pane"):
                        yield RichLog(id="config-log", markup=True, wrap=True)

        with Horizontal(id="status-bar"):
            yield Static("⬡ CPU: --", id="stat-cpu", classes="status-item")
            yield Static("▣ VRAM: --", id="stat-mem", classes="status-item")
            yield Static("⏱ 00:00:00", id="stat-uptime", classes="status-item")
            yield Static("↔ 0 turns", id="stat-turns", classes="status-item")
            yield Static("◉ looped inference", id="loop-badge")

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        log = self.query_one("#chat-log", RichLog)
        # Write welcome message
        log.write(
            f"[color={TEAL}]system[/]  "
            f"Ouro-2.6B-Thinking loaded  ·  "
            f"device: [bold]cuda:0[/]  ·  "
            f"recurrent steps: [bold color={PURPLE}]4[/]  ·  "
            f"context: 8192 tokens"
        )
        log.write("")
        
        # Start timers and updates
        self.set_interval(1.0, self._tick_uptime)
        self.set_interval(2.0, self._update_status)
        
        # Focus input
        self.query_one("#user-input", Input).focus()

    def _tick_uptime(self) -> None:
        """Update uptime counter."""
        self.uptime_seconds = int(time.time() - self._start_time)
        self._update_uptime_display()

    def _update_uptime_display(self) -> None:
        """Format and display uptime."""
        elapsed = self.uptime_seconds
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        uptime_str = f"{h:02d}:{m:02d}:{s:02d}"
        try:
            self.query_one("#stat-uptime", Static).update(f"⏱ {uptime_str}")
        except:
            pass  # Widget might not be ready yet

    def _update_status(self) -> None:
        """Update all status indicators."""
        try:
            self.query_one("#stat-turns", Static).update(f"↔ {self.turn_count} turns")
            
            # Add actual system monitoring if available
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.query_one("#stat-cpu", Static).update(f"⬡ CPU: {cpu_percent:.0f}%")
            
            # Try to get GPU memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    vram_used = torch.cuda.memory_allocated(0) / 1024**3
                    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    self.query_one("#stat-mem", Static).update(f"▣ VRAM: {vram_used:.1f}/{vram_total:.1f}GB")
                else:
                    self.query_one("#stat-mem", Static).update("▣ VRAM: CPU mode")
            except:
                self.query_one("#stat-mem", Static).update("▣ VRAM: --")
        except Exception as e:
            # Silently fail if widgets not ready
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        text = event.value.strip()
        if not text:
            return
        
        # Clear input field
        event.input.value = ""
        
        # Process message
        self._process_message(text)

    @work(thread=True)
    def _process_message(self, user_text: str) -> None:
        """Process user message in background thread."""
        # Get log widget safely
        def get_log():
            return self.query_one("#chat-log", RichLog)
        
        log = self.call_from_thread(get_log)
        ts = datetime.now().strftime("%H:%M:%S")

        # ── User line ──
        self.call_from_thread(
            log.write,
            f"[bold color={MUTED}]{ts}[/]  [bold color=#5a6090]you[/]  "
            f"[color=#8890c0]{user_text}[/]"
        )

        # ── Thinking indicator ──
        thinking_id = f"thinking_{time.time()}"
        self.call_from_thread(
            log.write,
            f"[dim]       ↻ thinking across loops...[/]",
            id=thinking_id
        )

        try:
            # ── Inference ──
            # Note: Make sure your agent.chat function is thread-safe
            response = agent.chat(self.history, user_text)
            
            # Update turn count on main thread
            self.call_from_thread(self._increment_turn_count)
            
            # Remove thinking indicator
            self.call_from_thread(log.remove_line, thinking_id)
            
            # ── Response ──
            ts2 = datetime.now().strftime("%H:%M:%S")
            self.call_from_thread(
                log.write,
                f"[bold color={MUTED}]{ts2}[/]  [bold color={TEAL}]ouro[/]  "
                f"[color={TEXT}]{response}[/]"
            )
            self.call_from_thread(log.write, "")
            
        except Exception as e:
            # Handle errors gracefully
            self.call_from_thread(log.remove_line, thinking_id)
            self.call_from_thread(
                log.write,
                f"[bold red]Error: {str(e)}[/]"
            )
        
        self.call_from_thread(self._update_status)

    def _increment_turn_count(self) -> None:
        """Increment turn count on main thread."""
        self.turn_count += 1

    def action_clear_chat(self) -> None:
        """Clear chat history."""
        self.query_one("#chat-log", RichLog).clear()
        # Reset history in agent module
        self.history.clear()
        self.turn_count = 0
        self._update_status()

    def action_save_session(self) -> None:
        """Save current session to file."""
        import json
        fname = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(fname, "w") as f:
                json.dump(self.history, f, indent=2)
            log = self.query_one("#chat-log", RichLog)
            log.write(f"[dim color={TEAL}]session saved → {fname}[/]")
        except Exception as e:
            log = self.query_one("#chat-log", RichLog)
            log.write(f"[dim red]Failed to save session: {e}[/]")

    def action_focus_tools(self) -> None:
        """Switch to tools tab."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "tools-pane"
        
    def action_focus_memory(self) -> None:
        """Switch to memory tab."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "memory-pane"

    def on_key(self, event) -> None:
        """Handle key events for tab switching."""
        if event.key == "f1":
            self.action_focus_tools()
        elif event.key == "f2":
            self.action_focus_memory()


if __name__ == "__main__":
    OuroApp().run()