from cx_Freeze import setup, Executable

build_options = {
    "include_files": ["favicon.ico", "app.png"],
}

exe = Executable(
    script="Extractor.py",
    base="Win32GUI",
    target_name="AudioExtractor.exe",
    icon="favicon.ico"
)

setup(
    name="AudioExtractor",
    version="1.0.0",
    description="Batch audio extractor GUI (FFmpeg)",
    options={"build_exe": build_options},
    executables=[exe]
)
