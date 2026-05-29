from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog, Static, TabbedContent, TabPane
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual import work
from textual.binding import Binding
from datetime import datetime
import time, agent

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
        width: 22;
        background: {BG_SIDE};
        border-right: tall {GHOST};
        height: 100%;
        padding: 0;
    }}
    #sidebar-title {{
        background: #10121e;
        border-bottom: tall {GHOST};
        padding: 1 2;
        color: {MUTED};
        text-style: bold;
        width: 100%;
    }}
    .nav-item {{
        padding: 0 2;
        color: {MUTED};
        height: 1;
    }}
    .nav-item:hover {{
        color: #8a90c0;
        background: #13162a;
    }}
    .nav-item.active {{
        color: {PURPLE};
        border-left: tall {PURPLE};
        background: #13162a;
    }}
    #model-info {{
        dock: bottom;
        border-top: tall {GHOST};
        padding: 1 2;
        color: {MUTED};
        background: {BG_SIDE};
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
        height: 100%;
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
        scrollbar-color-hover: {MUTED};
    }}

    /* ── Input bar ── */
    #input-area {{
        height: 5;
        border-top: tall {GHOST};
        background: {BG_INPUT};
        padding: 1 2;
    }}
    #user-input {{
        background: {BG_MAIN};
        border: tall {GHOST};
        color: {TEXT};
        width: 100%;
        height: 3;
    }}
    #user-input:focus {{
        border: tall {PURPLE_D};
    }}

    /* ── Status bar ── */
    #status-bar {{
        dock: bottom;
        height: 1;
        background: #090b18;
        border-top: tall {GHOST};
        color: {MUTED};
        padding: 0 2;
        layout: horizontal;
    }}
    .status-item {{
        margin-right: 2;
        color: {MUTED};
        height: 1;
    }}
    #loop-badge {{
        dock: right;
        color: {GHOST};
        height: 1;
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
    }}
    """

    BINDINGS = [
        Binding("ctrl+q", "quit",          "Quit",    show=True),
        Binding("ctrl+t", "focus_tools",   "Tools",   show=True),
        Binding("ctrl+m", "focus_memory",  "Memory",  show=True),
        Binding("ctrl+l", "clear_chat",    "Clear",   show=True),
        Binding("ctrl+s", "save_session",  "Save",    show=False),
    ]

    history: reactive[list] = reactive([])
    uptime_seconds: reactive[int] = reactive(0)
    turn_count: reactive[int] = reactive(0)

    def __init__(self):
        super().__init__()
        self._start_time = time.time()

    def compose(self) -> ComposeResult:
        with Horizontal(id="root-layout"):
            # ── Sidebar ──────────────────────────────────────────────
            with Vertical(id="sidebar"):
                yield Static("⟳ OURO AGENT", id="sidebar-title")
                yield Static("› chat",   classes="nav-item active")
                yield Static("  tools",  classes="nav-item")
                yield Static("  memory", classes="nav-item")
                yield Static("  config", classes="nav-item")
                with Static(id="model-info"):
                    yield Static("Ouro-2.6B-Thinking", id="model-name")
                    yield Static("steps: 4  float16", classes="status-item")
                    yield Static("ctx: 8192 tokens",  classes="status-item")

            # ── Main ─────────────────────────────────────────────────
            with Vertical(id="main-area"):
                with TabbedContent("Chat", "Tools", "Memory", "Config"):
                    with TabPane("Chat"):
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
                    with TabPane("Tools"):
                        yield RichLog(id="tools-log", markup=True, wrap=True)
                    with TabPane("Memory"):
                        yield RichLog(id="memory-log", markup=True, wrap=True)
                    with TabPane("Config"):
                        yield RichLog(id="config-log", markup=True, wrap=True)

        with Horizontal(id="status-bar"):
            yield Static(id="stat-cpu",    classes="status-item")
            yield Static(id="stat-mem",    classes="status-item")
            yield Static(id="stat-uptime", classes="status-item")
            yield Static(id="stat-turns",  classes="status-item")
            yield Static("◉ looped inference", id="loop-badge")

        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(
            f"[dim]{TEAL}]system[/]  "
            f"Ouro-2.6B-Thinking loaded  ·  "
            f"device: [bold]cuda:0[/]  ·  "
            f"recurrent steps: [bold #5050d0]4[/]  ·  "
            f"context: 8192 tokens"
        )
        log.write("")
        self.set_interval(1.0, self._tick_uptime)
        self._update_status()

    def _tick_uptime(self) -> None:
        self.uptime_seconds = int(time.time() - self._start_time)
        self._update_status()

    def _update_status(self) -> None:
        elapsed = self.uptime_seconds
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        uptime_str = f"{h:02d}:{m:02d}:{s:02d}"
        self.query_one("#stat-uptime", Static).update(f"⏱ {uptime_str}")
        self.query_one("#stat-turns",  Static).update(f"↔ {self.turn_count} turns")
        self.query_one("#stat-cpu",    Static).update("⬡ CPU")
        self.query_one("#stat-mem",    Static).update("▣ VRAM")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        self._process_message(text)

    @work(thread=True)
    def _process_message(self, user_text: str) -> None:
        log = self.call_from_thread(self.query_one, "#chat-log", RichLog)
        ts  = datetime.now().strftime("%H:%M:%S")

        # ── User line ──
        self.call_from_thread(
            log.write,
            f"[bold #3a4060]{ts}[/]  [bold #5a6090]you[/]  "
            f"[#8890c0]{user_text}[/]"
        )

        # ── Thinking indicator ──
        self.call_from_thread(
            log.write,
            f"[dim]       ↻ thinking across loops...[/]"
        )

        # ── Inference ──
        response = agent.chat(self.history, user_text)
        self.turn_count += 1

        # ── Response ──
        ts2 = datetime.now().strftime("%H:%M:%S")
        self.call_from_thread(
            log.write,
            f"[bold #3a4060]{ts2}[/]  [bold #207050]ouro[/]  "
            f"[#b0b8d8]{response}[/]"
        )
        self.call_from_thread(log.write, "")
        self.call_from_thread(self._update_status)

    def action_clear_chat(self) -> None:
        self.query_one("#chat-log", RichLog).clear()
        self.history = []
        self.turn_count = 0

    def action_save_session(self) -> None:
        import json
        fname = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fname, "w") as f:
            json.dump(self.history, f, indent=2)
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[dim]session saved → {fname}[/]")

    def action_focus_tools(self)  -> None: self.query_one(TabbedContent).active = "tab-2"
    def action_focus_memory(self) -> None: self.query_one(TabbedContent).active = "tab-3"


if __name__ == "__main__":
    OuroApp().run()