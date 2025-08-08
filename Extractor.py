"""
Audio Extractor
Creator: © Ardean Bima Saputra
- Mode: COPY, MP3, AAC (m4a), WAV
- Rekursif, preserve struktur folder
- Pilih audio track: by index (a:0) atau by language (eng/ind)
- Loudness normalization (EBU R128)
- Multithread + progress bar, log, Cancel
- Opsi GPU (CUDA) untuk mempercepat decoding video
- Preset: Optimasi Lagu + GPU
- Tanpa dependensi eksternal (pure stdlib). Butuh ffmpeg & ffprobe di PATH.
"""

import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, PhotoImage

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
DEFAULT_WORKERS = max(2, (os.cpu_count() or 4) // 2)

def resource_path(fname: str) -> str:
    """
    Cari file resource (ikon, dsb) di beragam mode bundle:
    - cx_Freeze: direktori exe
    - Nuitka onefile: env NUITKA_ONEFILE_TEMP_DIR
    - PyInstaller onefile: sys._MEIPASS
    - Mode script: direktori file .py
    """
    candidates = [
        os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else None,
        os.environ.get("NUITKA_ONEFILE_TEMP_DIR"),
        getattr(sys, "_MEIPASS", None),
        os.path.dirname(__file__),
    ]
    for base in candidates:
        if base:
            path = os.path.join(base, fname)
            if os.path.exists(path):
                return path
            
    return os.path.join(os.path.dirname(__file__), fname)

# ----------------- FFmpeg helpers -----------------

def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def ff_tools_ok() -> Tuple[bool, str]:
    for tool in ("ffmpeg", "ffprobe"):
        code, _, _ = run_cmd([tool, "-version"])
        if code != 0:
            return False, f"{tool} tidak ditemukan di PATH."
    return True, ""

def detect_audio_codec(file: Path, stream_selector: str) -> Optional[str]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", stream_selector,
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file)
    ]
    code, out, _ = run_cmd(cmd)
    if code == 0:
        val = out.strip().splitlines()
        if val:
            return val[0].strip()
    return None

def pick_copy_extension(codec: str) -> str:
    mapping = {
        "aac": ".m4a",
        "mp3": ".mp3",
        "ac3": ".ac3",
        "eac3": ".eac3",
        "dts": ".dts",
        "opus": ".opus",
        "vorbis": ".ogg",
        "flac": ".flac",
        "pcm_s16le": ".wav",
        "truehd": ".thd",
    }
    return mapping.get(codec, ".m4a")

def build_ffmpeg_cmd(
    src: Path,
    dst: Path,
    mode: str,
    stream_selector: str,
    loudnorm: bool,
    sample_rate: Optional[int],
    bitrate_k: Optional[int],
    use_gpu: bool = False
) -> List[str]:
    base = ["ffmpeg", "-y"]
    if use_gpu:
        base += ["-hwaccel", "cuda"]  # GPU decoding
    base += ["-i", str(src), "-vn", "-sn", "-dn", "-map", f"0:{stream_selector}"]

    af = []
    if loudnorm:
        af.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    if mode == "COPY":
        base += ["-c:a", "copy"]
    elif mode == "MP3":
        base += ["-c:a", "libmp3lame"]
        if bitrate_k:
            base += ["-b:a", f"{bitrate_k}k"]
        if sample_rate:
            base += ["-ar", str(sample_rate)]
    elif mode == "AAC":
        base += ["-c:a", "aac"]
        if bitrate_k:
            base += ["-b:a", f"{bitrate_k}k"]
        else:
            base += ["-q:a", "2"]
        if sample_rate:
            base += ["-ar", str(sample_rate)]
    elif mode == "WAV":
        base += ["-c:a", "pcm_s16le", "-ac", "2"]
        if sample_rate:
            base += ["-ar", str(sample_rate)]

    if af:
        base += ["-af", ",".join(af)]

    base += [str(dst)]
    return base

# ----------------- Worker logic -----------------

def scan_files(input_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        return [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return [p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]

def make_out_path(
    src: Path, input_root: Path, output_root: Path, preserve_tree: bool,
    mode: str, stream_selector: str
) -> Path:
    if preserve_tree:
        rel = src.relative_to(input_root)
        stem = rel.with_suffix("")
    else:
        stem = Path(src.stem)

    if mode == "COPY":
        codec = detect_audio_codec(src, stream_selector)
        ext = pick_copy_extension(codec or "aac")
    elif mode == "MP3":
        ext = ".mp3"
    elif mode == "AAC":
        ext = ".m4a"
    else:
        ext = ".wav"

    out_path = (output_root / stem).with_suffix(ext)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path

def process_one(
    src: Path, input_root: Path, output_root: Path, preserve_tree: bool,
    mode: str, stream_selector: str, loudnorm: bool,
    sample_rate: Optional[int], bitrate_k: Optional[int],
    use_gpu: bool = False
) -> Tuple[Path, bool, str]:
    try:
        dst = make_out_path(src, input_root, output_root, preserve_tree, mode, stream_selector)
        cmd = build_ffmpeg_cmd(src, dst, mode, stream_selector, loudnorm, sample_rate, bitrate_k, use_gpu)
        code, _, err = run_cmd(cmd)
        if code != 0:
            return src, False, err.strip()
        return src, True, str(dst)
    except Exception as e:
        return src, False, str(e)

# ----------------- GUI -----------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # ==== SET ICON ====
        png_path = resource_path("app.png")         # ikon window (disarankan PNG)
        ico_path = resource_path("favicon.ico")     # fallback ICO (Windows)

        set_ok = False
        try:
            if os.path.exists(png_path):
                self.iconphoto(False, PhotoImage(file=png_path))
                set_ok = True
        except Exception as e:
            print(f"[Icon] PNG gagal: {e}")

        if not set_ok:
            try:
                if os.path.exists(ico_path):
                    self.iconbitmap(ico_path)
                    set_ok = True
            except Exception as e:
                print(f"[Icon] ICO gagal: {e}")

        if not set_ok:
            print("[Icon] Ikon tidak dipasang, lanjut dengan ikon default.")
            
        self.iconbitmap(ico_path)
        self.title("Audio Extractor")
        self.geometry("860x660")
        self.minsize(860, 660)

        self.stop_event = threading.Event()
        self.ui_queue = queue.Queue()

        self._build_ui()
        self._after_poll()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # Paths
        frm_paths = ttk.LabelFrame(self, text="Lokasi")
        frm_paths.pack(fill="x", **pad)

        self.in_var = tk.StringVar()
        self.out_var = tk.StringVar()

        ttk.Label(frm_paths, text="Input Folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm_paths, textvariable=self.in_var, width=80).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(frm_paths, text="Browse", command=self.pick_in).grid(row=0, column=2)

        ttk.Label(frm_paths, text="Output Folder").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm_paths, textvariable=self.out_var, width=80).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(frm_paths, text="Browse", command=self.pick_out).grid(row=1, column=2)

        frm_paths.columnconfigure(1, weight=1)

        # Options
        frm_opts = ttk.LabelFrame(self, text="Opsi")
        frm_opts.pack(fill="x", **pad)

        ttk.Label(frm_opts, text="Mode").grid(row=0, column=0, sticky="w")
        self.mode_var = tk.StringVar(value="COPY")
        ttk.Combobox(frm_opts, textvariable=self.mode_var, values=["COPY", "MP3", "AAC", "WAV"], state="readonly", width=10).grid(row=0, column=1, sticky="w")

        self.recursive_var = tk.BooleanVar(value=True)
        self.preserve_var = tk.BooleanVar(value=True)
        self.gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_opts, text="Rekursif", variable=self.recursive_var).grid(row=0, column=2, sticky="w", padx=12)
        ttk.Checkbutton(frm_opts, text="Pertahankan struktur folder", variable=self.preserve_var).grid(row=0, column=3, sticky="w", padx=12)
        ttk.Checkbutton(frm_opts, text="Gunakan GPU (CUDA)", variable=self.gpu_var).grid(row=0, column=4, sticky="w", padx=12)

        # Stream selector
        self.sel_mode_var = tk.StringVar(value="index")
        frm_stream = ttk.Frame(frm_opts)
        frm_stream.grid(row=1, column=0, columnspan=5, sticky="w", pady=4)
        ttk.Label(frm_stream, text="Audio Track:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(frm_stream, text="Index", variable=self.sel_mode_var, value="index").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(frm_stream, text="Bahasa (ISO-639-2)", variable=self.sel_mode_var, value="lang").grid(row=0, column=2, sticky="w")

        self.index_var = tk.StringVar(value="0")
        self.lang_var  = tk.StringVar(value="eng")
        ttk.Label(frm_stream, text="Index").grid(row=0, column=3, sticky="e", padx=(18,4))
        ttk.Entry(frm_stream, textvariable=self.index_var, width=6).grid(row=0, column=4, sticky="w")
        ttk.Label(frm_stream, text="Bahasa").grid(row=0, column=5, sticky="e", padx=(18,4))
        ttk.Entry(frm_stream, textvariable=self.lang_var, width=6).grid(row=0, column=6, sticky="w")

        # Loudness + SR + bitrate + workers
        self.loud_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_opts, text="Loudness normalization (EBU R128)", variable=self.loud_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6,0))

        ttk.Label(frm_opts, text="Sample rate (Hz)").grid(row=2, column=2, sticky="e")
        self.sr_var = tk.StringVar(value="")
        ttk.Entry(frm_opts, textvariable=self.sr_var, width=10).grid(row=2, column=3, sticky="w", padx=6)

        ttk.Label(frm_opts, text="Bitrate (kbps)").grid(row=3, column=2, sticky="e")
        self.br_var = tk.StringVar(value="")
        ttk.Entry(frm_opts, textvariable=self.br_var, width=10).grid(row=3, column=3, sticky="w", padx=6)

        ttk.Label(frm_opts, text="Workers").grid(row=3, column=0, sticky="e")
        self.workers_var = tk.StringVar(value=str(DEFAULT_WORKERS))
        ttk.Entry(frm_opts, textvariable=self.workers_var, width=6).grid(row=3, column=1, sticky="w")

        # Actions
        frm_actions = ttk.Frame(self)
        frm_actions.pack(fill="x", **pad)

        self.btn_start = ttk.Button(frm_actions, text="Start", command=self.start)
        self.btn_cancel = ttk.Button(frm_actions, text="Cancel", command=self.cancel, state="disabled")
        self.btn_preset_music_gpu = ttk.Button(frm_actions, text="Preset: Optimasi Lagu + GPU", command=self.apply_preset_music_gpu)  # NEW
        self.btn_start.pack(side="left")
        self.btn_cancel.pack(side="left", padx=8)
        self.btn_preset_music_gpu.pack(side="right")  # put it on the right for quick access

        # Progress
        frm_prog = ttk.LabelFrame(self, text="Progress")
        frm_prog.pack(fill="x", **pad)
        self.progress = ttk.Progressbar(frm_prog, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=8, pady=8)

        self.status_var = tk.StringVar(value="Menunggu...")
        ttk.Label(frm_prog, textvariable=self.status_var).pack(anchor="w", padx=8, pady=(0,8))

        # Log
        frm_log = ttk.LabelFrame(self, text="Log")
        frm_log.pack(fill="both", expand=True, **pad)
        self.log_widget = tk.Text(frm_log, height=16, wrap="none")
        self.log_widget.pack(fill="both", expand=True, padx=8, pady=8)
        self.log_widget.configure(state="disabled")

    # Preset implementasi
    def apply_preset_music_gpu(self):
        # Mode MP3
        self.mode_var.set("MP3")
        # Loudness ON
        self.loud_var.set(True)
        # Bitrate 320
        self.br_var.set("320")
        # Sample rate 44100
        self.sr_var.set("44100")
        # GPU ON
        self.gpu_var.set(True)
        # Workers: minimal 4, maksimal 6, tanpa melebihi DEFAULT_WORKERS
        target_workers = min(6, max(4, DEFAULT_WORKERS))
        self.workers_var.set(str(target_workers))
        # Rekursif OFF (biar fokus ke folder itu saja)
        self.recursive_var.set(False)
        # Track index 0
        self.sel_mode_var.set("index")
        self.index_var.set("0")
        # Status kecil manis
        self.set_status("Preset 'Optimasi Lagu + GPU' diterapkan.")

    # UI helpers
    def pick_in(self):
        path = filedialog.askdirectory(title="Pilih input folder")
        if path:
            self.in_var.set(path)

    def pick_out(self):
        path = filedialog.askdirectory(title="Pilih output folder")
        if path:
            self.out_var.set(path)

    def log(self, text: str):
        self.ui_queue.put(("log", text))

    def set_status(self, text: str):
        self.ui_queue.put(("status", text))

    def set_progress(self, value: int, total: int):
        self.ui_queue.put(("progress", (value, total)))

    def _after_poll(self):
        try:
            while True:
                kind, data = self.ui_queue.get_nowait()
                if kind == "log":
                    self.log_widget.configure(state="normal")
                    self.log_widget.insert("end", data + "\n")
                    self.log_widget.see("end")
                    self.log_widget.configure(state="disabled")
                elif kind == "status":
                    self.status_var.set(data)
                elif kind == "progress":
                    val, total = data
                    self.progress["maximum"] = max(1, total)
                    self.progress["value"] = val
        except queue.Empty:
            pass
        self.after(50, self._after_poll)

    def start(self):
        ok, msg = ff_tools_ok()
        if not ok:
            messagebox.showerror("Error", msg + "\nInstall FFmpeg dan pastikan PATH benar.")
            return

        input_dir = Path(self.in_var.get() or ".").resolve()
        output_dir = Path(self.out_var.get() or (Path.cwd() / "audio_out")).resolve()
        recursive = self.recursive_var.get()
        preserve = self.preserve_var.get()
        mode = self.mode_var.get()

        sel_mode = self.sel_mode_var.get()
        if sel_mode == "index":
            try:
                idx = int(self.index_var.get().strip())
                stream_selector = f"a:{idx}"
            except ValueError:
                messagebox.showerror("Error", "Index audio harus angka.")
                return
        else:
            lang = self.lang_var.get().strip() or "eng"
            stream_selector = f"a:m:language:{lang}"

        loud = self.loud_var.get()

        sr = self.sr_var.get().strip()
        sample_rate = int(sr) if sr else None

        br = self.br_var.get().strip()
        bitrate_k = int(br) if br else None

        try:
            workers = max(1, int(self.workers_var.get().strip()))
        except ValueError:
            messagebox.showerror("Error", "Workers harus angka.")
            return

        if not input_dir.exists():
            messagebox.showerror("Error", "Input folder tidak valid.")
            return
        output_dir.mkdir(parents=True, exist_ok=True)

        files = scan_files(input_dir, recursive)
        if not files:
            messagebox.showinfo("Info", "Tidak ada file video ditemukan.")
            return

        # lock UI
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.btn_preset_music_gpu.config(state="disabled")
        self.stop_event.clear()
        self.progress["value"] = 0
        self.progress["maximum"] = len(files)
        self.set_status(f"Ditemukan {len(files)} file. Memulai...")

        args = dict(
            input_root=input_dir,
            output_root=output_dir,
            preserve_tree=preserve,
            mode=mode,
            stream_selector=stream_selector,
            loudnorm=loud,
            sample_rate=sample_rate,
            bitrate_k=bitrate_k,
            use_gpu=self.gpu_var.get(),
        )

        threading.Thread(target=self._run_batch, args=(files, args, workers), daemon=True).start()

    def cancel(self):
        self.stop_event.set()
        self.set_status("Membatalkan...")

    def _run_batch(self, files: List[Path], args: dict, workers: int):
        ok_count = 0
        fail_count = 0
        done = 0
        total = len(files)

        def task(f: Path):
            if self.stop_event.is_set():
                return f, False, "Dibatalkan."
            return process_one(
                f,
                args["input_root"],
                args["output_root"],
                args["preserve_tree"],
                args["mode"],
                args["stream_selector"],
                args["loudnorm"],
                args["sample_rate"],
                args["bitrate_k"],
                args["use_gpu"]
            )

        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(task, f) for f in files]
                for fut in as_completed(futures):
                    if self.stop_event.is_set():
                        break
                    src, success, msg = fut.result()
                    done += 1
                    if success:
                        ok_count += 1
                        self.log(f"[OK] {src.name} -> {msg}")
                    else:
                        fail_count += 1
                        self.log(f"[FAIL] {src.name} :: {msg}")
                    self.set_progress(done, total)
                    self.set_status(f"Proses: {done}/{total} | OK: {ok_count} | Gagal: {fail_count}")

        except Exception as e:
            self.log(f"[ERROR] {e}")

        finally:
            self.btn_start.config(state="normal")
            self.btn_cancel.config(state="disabled")
            self.btn_preset_music_gpu.config(state="normal")
            if self.stop_event.is_set():
                self.set_status(f"Dibatalkan. OK: {ok_count}, Gagal: {fail_count}")
                self.log("== Dibatalkan ==")
            else:
                self.set_status(f"Selesai. OK: {ok_count}, Gagal: {fail_count}")
                self.log("== Selesai ==")

if __name__ == "__main__":
    app = App()
    app.mainloop()
