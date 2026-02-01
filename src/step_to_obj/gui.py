"""GUI for STEP converter with drag & drop support."""

import threading
from pathlib import Path
from tkinter import filedialog, StringVar
from tkinterdnd2 import DND_FILES, TkinterDnD
import customtkinter as ctk

from .converter import convert_step, ConversionResult, OutputFormat


class DropZone(ctk.CTkFrame):
    """Drag and drop zone for STEP files."""

    def __init__(self, master, on_file_drop: callable, **kwargs):
        super().__init__(master, **kwargs)
        self.on_file_drop = on_file_drop
        self.is_dragging = False

        self.configure(
            fg_color=("#e8e8e8", "#2b2b2b"),
            corner_radius=12,
            border_width=2,
            border_color=("#cccccc", "#404040")
        )

        # Icon label
        self.icon_label = ctk.CTkLabel(
            self,
            text="📁",
            font=ctk.CTkFont(size=48)
        )
        self.icon_label.pack(pady=(30, 10))

        # Main text
        self.text_label = ctk.CTkLabel(
            self,
            text="STEPファイルをここにドロップ",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.text_label.pack(pady=5)

        # Sub text
        self.sub_label = ctk.CTkLabel(
            self,
            text="または クリックしてファイルを選択",
            font=ctk.CTkFont(size=12),
            text_color=("#666666", "#999999")
        )
        self.sub_label.pack(pady=(0, 30))

        # Bind click to open file dialog
        self.bind("<Button-1>", self._on_click)
        self.icon_label.bind("<Button-1>", self._on_click)
        self.text_label.bind("<Button-1>", self._on_click)
        self.sub_label.bind("<Button-1>", self._on_click)

    def _on_click(self, event):
        """Open file dialog on click."""
        file_path = filedialog.askopenfilename(
            title="STEPファイルを選択",
            filetypes=[
                ("STEP files", "*.step *.stp *.STEP *.STP"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.on_file_drop(file_path)

    def set_drag_state(self, is_dragging: bool):
        """Update visual state for drag events."""
        self.is_dragging = is_dragging
        if is_dragging:
            self.configure(
                border_color=("#3b82f6", "#3b82f6"),
                fg_color=("#dbeafe", "#1e3a5f")
            )
        else:
            self.configure(
                border_color=("#cccccc", "#404040"),
                fg_color=("#e8e8e8", "#2b2b2b")
            )


class FileInfoFrame(ctk.CTkFrame):
    """Display selected file information."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")

        self.file_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=13),
            anchor="w"
        )
        self.file_label.pack(fill="x", padx=10, pady=5)

    def set_file(self, file_path: Path | None):
        """Update displayed file information."""
        if file_path:
            self.file_label.configure(
                text=f"選択中: {file_path.name}"
            )
        else:
            self.file_label.configure(text="")


class FormatSelectorFrame(ctk.CTkFrame):
    """Output format selector."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=5)

        label = ctk.CTkLabel(
            row,
            text="形式:",
            font=ctk.CTkFont(size=13),
            width=60
        )
        label.pack(side="left")

        self.format_var = StringVar(value="OBJ")
        self.format_combo = ctk.CTkComboBox(
            row,
            values=["OBJ", "FBX"],
            variable=self.format_var,
            state="readonly",
            width=120,
            font=ctk.CTkFont(size=13)
        )
        self.format_combo.pack(side="left", padx=(10, 0))

    def get_format(self) -> OutputFormat:
        """Get selected output format."""
        value = self.format_var.get()
        if value == "FBX":
            return OutputFormat.FBX
        return OutputFormat.OBJ


class OutputSettingsFrame(ctk.CTkFrame):
    """Output directory settings."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")
        self.output_dir: Path | None = None

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=5)

        label = ctk.CTkLabel(
            row,
            text="出力先:",
            font=ctk.CTkFont(size=13),
            width=60
        )
        label.pack(side="left")

        self.path_var = StringVar(value="(入力ファイルと同じ場所)")
        self.path_entry = ctk.CTkEntry(
            row,
            textvariable=self.path_var,
            state="readonly",
            font=ctk.CTkFont(size=12)
        )
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(10, 10))

        self.browse_btn = ctk.CTkButton(
            row,
            text="変更",
            width=60,
            command=self._browse_output
        )
        self.browse_btn.pack(side="right")

    def _browse_output(self):
        """Open directory selection dialog."""
        dir_path = filedialog.askdirectory(title="出力先フォルダを選択")
        if dir_path:
            self.output_dir = Path(dir_path)
            self.path_var.set(str(self.output_dir))

    def get_output_dir(self) -> Path | None:
        """Get selected output directory."""
        return self.output_dir


class ConvertButton(ctk.CTkButton):
    """Styled convert button."""

    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            text="変換開始",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=45,
            corner_radius=8,
            **kwargs
        )


class StatusBar(ctk.CTkFrame):
    """Status bar for displaying conversion progress and results."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")

        self.status_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(pady=5)

        self.progress = ctk.CTkProgressBar(self)
        self.progress.pack(fill="x", padx=20, pady=5)
        self.progress.set(0)
        self.progress.pack_forget()

    def set_status(self, message: str, is_error: bool = False):
        """Update status message."""
        color = ("#dc2626", "#ef4444") if is_error else ("#16a34a", "#22c55e")
        if not message:
            color = ("#666666", "#999999")
        self.status_label.configure(text=message, text_color=color)

    def show_progress(self, show: bool = True):
        """Show or hide progress bar."""
        if show:
            self.progress.pack(fill="x", padx=20, pady=5)
            self.progress.configure(mode="indeterminate")
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.pack_forget()


class App(TkinterDnD.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title("STEP Converter")
        # Increase height to accommodate status bar and padding
        self.geometry("500x550")
        self.minsize(450, 500)

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.selected_file: Path | None = None

        self._create_widgets()
        self._setup_dnd()

    def _create_widgets(self):
        """Create all UI widgets."""
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="STEP → OBJ / FBX 変換",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.title_label.pack(pady=(0, 15))

        # Drop zone
        self.drop_zone = DropZone(
            self.main_frame,
            on_file_drop=self._on_file_selected,
            height=150
        )
        self.drop_zone.pack(fill="x", pady=(0, 10))

        # File info
        self.file_info = FileInfoFrame(self.main_frame)
        self.file_info.pack(fill="x")

        # Format selector
        self.format_selector = FormatSelectorFrame(self.main_frame)
        self.format_selector.pack(fill="x", pady=5)

        # Output settings
        self.output_settings = OutputSettingsFrame(self.main_frame)
        self.output_settings.pack(fill="x", pady=5)

        # Convert button
        self.convert_btn = ConvertButton(
            self.main_frame,
            command=self._start_conversion,
            state="disabled"
        )
        self.convert_btn.pack(fill="x", padx=10, pady=10)

        # Status bar
        self.status_bar = StatusBar(self.main_frame)
        self.status_bar.pack(fill="x", pady=(10, 0))

    def _setup_dnd(self):
        """Setup drag and drop handlers."""
        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<DropEnter>>", self._on_drag_enter)
        self.dnd_bind("<<DropLeave>>", self._on_drag_leave)
        self.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drag_enter(self, event):
        """Handle drag enter event."""
        self.drop_zone.set_drag_state(True)

    def _on_drag_leave(self, event):
        """Handle drag leave event."""
        self.drop_zone.set_drag_state(False)

    def _on_drop(self, event):
        """Handle file drop event."""
        self.drop_zone.set_drag_state(False)

        file_path = event.data.strip()
        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]

        if " " in file_path and not Path(file_path).exists():
            file_path = file_path.split()[0]

        self._on_file_selected(file_path)

    def _on_file_selected(self, file_path: str):
        """Handle file selection."""
        path = Path(file_path)

        if path.suffix.lower() not in (".step", ".stp"):
            self.status_bar.set_status("STEPファイル (.step, .stp) を選択してください", is_error=True)
            return

        if not path.exists():
            self.status_bar.set_status("ファイルが見つかりません", is_error=True)
            return

        self.selected_file = path
        self.file_info.set_file(path)
        self.convert_btn.configure(state="normal")
        self.status_bar.set_status("")

    def _start_conversion(self):
        """Start the conversion process in a background thread."""
        if not self.selected_file:
            return

        output_format = self.format_selector.get_format()
        output_dir = self.output_settings.get_output_dir()
        if output_dir is None:
            output_dir = self.selected_file.parent

        extension = "." + output_format.value
        output_path = output_dir / (self.selected_file.stem + extension)

        self.convert_btn.configure(state="disabled", text="変換中...")
        self.status_bar.show_progress(True)
        self.status_bar.set_status("変換中...")

        def run_conversion():
            result = convert_step(self.selected_file, output_path, output_format)
            self.after(0, lambda: self._on_conversion_complete(result))

        thread = threading.Thread(target=run_conversion, daemon=True)
        thread.start()

    def _on_conversion_complete(self, result: ConversionResult):
        """Handle conversion completion."""
        self.convert_btn.configure(state="normal", text="変換開始")
        self.status_bar.show_progress(False)

        if result.success:
            self.status_bar.set_status(
                f"完了: {result.vertex_count:,} 頂点, {result.face_count:,} 面"
            )
        else:
            self.status_bar.set_status(result.message, is_error=True)


def run_app():
    """Run the application."""
    app = App()
    app.mainloop()
