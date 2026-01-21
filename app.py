import os
import logging
import sys
import json
import gradio as gr
from dotenv import load_dotenv

from db.db import init_db, create_job, get_recent_jobs, get_job, get_job_artifacts
from worker.generator import run_generation_job
from worker.system_check import (
    check_cuda_status,
    check_model_status,
    get_system_status_text,
    is_ready_to_generate,
)
from worker.model_downloader import (
    download_all_models_generator,
    get_model_info_text,
    get_download_state,
    check_huggingface_hub,
)

# Load .env
load_dotenv()


# --- Configuration ---
class Config:
    BASE_DIR = os.getcwd()

    # Paths
    MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models"))
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "outputs"))
    DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "beatbunny.db"))
    LOG_DIR = os.getenv("LOG_DIR", os.path.join(BASE_DIR, "logs"))
    LOG_FILE = os.path.join(LOG_DIR, "beatbunny.log")
    PRESETS_FILE = os.path.join(BASE_DIR, "presets", "tags.json")

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        for path in [
            cls.MODEL_DIR,
            cls.OUTPUT_DIR,
            cls.LOG_DIR,
            os.path.dirname(cls.DB_PATH),
        ]:
            if path:
                os.makedirs(path, exist_ok=True)


# --- Logging Setup ---
def setup_logging():
    """Configures logging to both file and console."""
    Config.ensure_directories()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(Config.LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info("Logging initialized.")
    logging.info(f"Outputs: {Config.OUTPUT_DIR}")
    logging.info(f"Database: {Config.DB_PATH}")


setup_logging()
init_db(Config.DB_PATH)

# --- Helper Functions ---


def load_presets():
    """Load tag presets from JSON."""
    if os.path.exists(Config.PRESETS_FILE):
        try:
            with open(Config.PRESETS_FILE, "r") as f:
                data = json.load(f)
                # Flatten lists for simple dropdown
                all_tags = []
                for category in data.values():
                    all_tags.extend(category)
                # Convert all tags to lowercase (HeartMuLa requirement)
                return sorted([tag.lower() for tag in all_tags])
        except Exception as e:
            logging.error(f"Failed to load presets: {e}")
    return []


PRESET_TAGS = load_presets()


def get_template_lyrics():
    return """[Verse 1]
(Write your verse here...)

[Chorus]
(Write your chorus here...)

[Verse 2]
(Write your second verse here...)

[Chorus]
(Repeat chorus...)

[Outro]
(Fade out...)"""


def format_history_row(job):
    """Format a job row for the history listbox."""
    # job is a dict from sqlite3.Row
    status_icon = (
        "🟢"
        if job["status"] == "completed"
        else "🔴"
        if job["status"] == "failed"
        else "⏳"
    )
    return f"{status_icon} {job['created_at'][:19]} | {job['id'][:8]}..."


def load_history_list():
    """Get list of recent jobs formatted for display."""
    jobs = get_recent_jobs(20)
    # Return list of tuples (label, value) for Dropdown/Radio, or just list of strings
    # For a listbox-like experience, we can use a helper to look up ID from string
    return [format_history_row(j) for j in jobs]


def get_job_details(history_selection):
    """Load details when a history item is clicked."""
    if not history_selection:
        return None, None, None, None, None

    # Extract ID from string "🟢 2023-... | abc1234..."
    try:
        job_id_short = history_selection.split("|")[1].strip().replace("...", "")
        # This is a weak lookup (short ID), ideally we store full ID in value
        # But get_recent_jobs returns full rows, let's just find the match in DB
        # Re-querying by partial ID is risky, let's fetch recent again to map
        jobs = get_recent_jobs(20)
        selected_job = next((j for j in jobs if j["id"].startswith(job_id_short)), None)

        if not selected_job:
            return None, None, None, None, None

        job_id = selected_job["id"]
        job = get_job(job_id)
        if not job:
            return None, None, "Error: Job not found in DB", None, None

        artifacts = get_job_artifacts(job_id)

        # Collect artifacts for download
        download_paths = []
        audio_path = None

        for art in artifacts:
            path = art["path"]
            if os.path.exists(path):
                download_paths.append(path)
                # Prefer wav for audio player, or mp3 if wav missing
                if art["type"] == "audio_wav":
                    audio_path = path
                elif art["type"] == "audio_mp3" and not audio_path:
                    audio_path = path

        # Parse params
        params_str = job["params"]
        # Format info text
        info_text = (
            f"Status: {job['status']}\nID: {job['id']}\nCreated: {job['created_at']}\n"
        )
        if job["error"]:
            info_text += f"\nERROR: {job['error']}"

        return (
            job["lyrics"],
            job["tags"],
            info_text,
            audio_path,
            gr.update(value=download_paths, visible=bool(download_paths)),
        )

    except Exception as e:
        logging.error(f"Error loading job details: {e}")
        return None, None, f"Error: {e}", None, None


def generate_music(lyrics, tags, cfg, temp, top_k, length, seed):
    """
    Main generation handler.
    1. Create Job in DB
    2. Run generation
    3. Return results
    """
    try:
        # Input validation
        if not lyrics or not lyrics.strip():
            yield "❌ Error: Lyrics cannot be empty.", None, gr.update(visible=False)
            return
        if not tags or not tags.strip():
            yield "❌ Error: Tags cannot be empty.", None, gr.update(visible=False)
            return

        # Log inputs for debugging
        logging.info(f"Generate: Lyrics length={len(lyrics)}, Tags={tags}")
        logging.info(f"Generate: Lyrics preview: {lyrics[:100]}...")

        # 1. Prepare params
        params = {
            "cfg_scale": cfg,
            "temperature": temp,
            "top_k": top_k,
            "max_length": length,
            "seed": seed,
        }

        # 2. Create Job
        job_id = create_job(lyrics, tags, params)
        logging.info(f"Created job {job_id}")

        # 3. Run (Synchronous for MVP0)
        status_msg = f"Job {job_id} running... (Length: {length}s)"
        yield status_msg, None, gr.update(visible=False)

        result = run_generation_job(job_id, lyrics, tags, params, Config)

        if result["success"]:
            msg = f"✅ Success! Job {job_id} completed."
            audio_file = result["audio_path"]
            job_dir = os.path.dirname(audio_file)
            
            # Collect all downloadable artifacts (WAV, MP3, metadata.json)
            dl_files = [audio_file]
            mp3_path = os.path.splitext(audio_file)[0] + ".mp3"
            metadata_path = os.path.join(job_dir, "metadata.json")
            
            if os.path.exists(mp3_path):
                dl_files.append(mp3_path)
            if os.path.exists(metadata_path):
                dl_files.append(metadata_path)

            yield msg, audio_file, gr.update(value=dl_files, visible=True)
        else:
            msg = f"❌ Failed: {result.get('error', 'Unknown error')}"
            yield msg, None, gr.update(visible=False)

    except Exception as e:
        logging.error(f"Generate handler failed: {e}")
        msg = f"❌ Error: {str(e)}"
        yield msg, None, gr.update(visible=False)


# --- UI Styling ---

CUSTOM_CSS = """
/* Background: Deep animated gradient */
body, .gradio-container {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    color: white !important;
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glassmorphism Panels */
.glass-panel {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
}

/* Headers */
h1, h2, h3 {
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Inputs & Textboxes */
textarea, input {
    background-color: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #eee !important;
}

/* Buttons */
button.primary {
    background: linear-gradient(90deg, #ff00cc, #333399) !important;
    border: none !important;
    color: white !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    box-shadow: 0 0 15px rgba(255, 0, 204, 0.4) !important;
    transition: all 0.3s ease;
}
button.primary:hover {
    box-shadow: 0 0 25px rgba(255, 0, 204, 0.7) !important;
    transform: translateY(-2px);
}

/* Audio Player */
audio {
    width: 100%;
    filter: sepia(20%) hue-rotate(280deg) saturate(200%);
}
"""

# Theme: Dark mode with Fuchsia accents
theme = gr.themes.Soft(
    primary_hue="fuchsia",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="transparent",
    block_background_fill="transparent",
    block_border_width="0px",
)

# --- UI Definition ---

def get_initial_status():
    """Get initial system status for display."""
    return get_system_status_text(Config.MODEL_DIR)

def refresh_system_status():
    """Refresh system status display."""
    return get_system_status_text(Config.MODEL_DIR)

def get_readiness_banner():
    """Get a readiness status banner for the generate tab."""
    ready, message = is_ready_to_generate(Config.MODEL_DIR)
    if ready:
        return gr.update(value=message, visible=False)
    return gr.update(value=message, visible=True)

def run_model_download():
    """Generator function for model download with streaming output."""
    yield "Starting download process...\n"
    for msg in download_all_models_generator(Config.MODEL_DIR):
        yield msg

with gr.Blocks(title="BeatBunny MVP0", theme=theme, css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🐇 BeatBunny `Studio`")
    
    with gr.Tabs():
        # ============ GENERATE TAB ============
        with gr.Tab("🎵 Generate", id="generate-tab"):
            # Readiness warning banner
            readiness_banner = gr.Textbox(
                value="",
                visible=False,
                interactive=False,
                show_label=False,
                elem_id="readiness-banner"
            )
            
            with gr.Row():
                # LEFT COLUMN: Inputs
                with gr.Column(scale=2, elem_classes="glass-panel"):
                    with gr.Row():
                        gr.Markdown("### 1. Lyrics")
                        template_btn = gr.Button(
                            "Insert Template", size="sm", variant="secondary"
                        )

                    lyrics_input = gr.Textbox(
                        label="Lyrics (Text)",
                        placeholder="Enter lyrics here... use [Verse], [Chorus] headers.",
                        lines=10,
                        elem_id="lyrics-box",
                    )

                    gr.Markdown("### 2. Style & Tags")
                    gr.Markdown("💡 **Tip:** Use lowercase tags. Include `male vocals` or `female vocals` to control singer gender. Format: comma-separated, no spaces.")
                    with gr.Row():
                        tags_input = gr.Textbox(
                            label="Tags", placeholder="rock,upbeat,male vocals,guitar", scale=3
                        )
                        preset_dropdown = gr.Dropdown(
                            label="Presets", choices=PRESET_TAGS, scale=1
                        )

                    # Helper to append preset to tags
                    def add_preset(current_tags, preset):
                        if not preset:
                            return current_tags
                        if current_tags:
                            return f"{current_tags}, {preset}"
                        return preset

                    preset_dropdown.change(
                        add_preset, [tags_input, preset_dropdown], tags_input
                    )
                    template_btn.click(get_template_lyrics, outputs=lyrics_input)

                    gr.Markdown("### 3. Parameters")
                    with gr.Group():
                        with gr.Row():
                            cfg_slider = gr.Slider(
                                label="CFG Scale (Guidance)",
                                info="HeartMuLa default: 1.5. Higher = stronger conditioning.",
                                minimum=1.0,
                                maximum=10.0,
                                value=1.5,
                                step=0.1,
                            )
                            temp_slider = gr.Slider(
                                label="Temperature",
                                info="HeartMuLa default: 1.0. Lower = more deterministic.",
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                            )
                        with gr.Row():
                            top_k_slider = gr.Slider(
                                label="Top-K Sampling",
                                info="HeartMuLa default: 50. Higher = more diversity.",
                                minimum=10,
                                maximum=250,
                                value=50,
                                step=10,
                            )
                            length_dropdown = gr.Dropdown(
                                label="Duration (seconds)", choices=[30, 60, 120, 240], value=30
                            )
                        seed_input = gr.Number(label="Seed (Optional)", precision=0, value=None)

                    generate_btn = gr.Button("🎵 Generate Music", variant="primary", size="lg")

                # RIGHT COLUMN: Outputs & History
                with gr.Column(scale=2, elem_classes="glass-panel"):
                    gr.Markdown("### Status & Output")
                    status_box = gr.Textbox(label="Status", value="Ready", interactive=False)

                    audio_output = gr.Audio(label="Generated Audio", type="filepath")
                    download_files = gr.File(
                        label="Download Artifacts", visible=False, file_count="multiple"
                    )

                    gr.Markdown("---")

                    with gr.Row():
                        gr.Markdown("### History")
                        refresh_hist_btn = gr.Button("🔄", size="sm")

                    history_list = gr.Dropdown(
                        label="Recent Jobs (Select to load)",
                        choices=load_history_list(),
                        interactive=True,
                    )

        # ============ SETUP TAB ============
        with gr.Tab("⚙️ Setup", id="setup-tab"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="glass-panel"):
                    gr.Markdown("## System Status")
                    
                    # System status display
                    system_status_md = gr.Markdown(
                        value=get_initial_status,
                        label="System Status"
                    )
                    
                    refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                    refresh_status_btn.click(
                        fn=refresh_system_status,
                        outputs=system_status_md
                    )
                    
                with gr.Column(scale=1, elem_classes="glass-panel"):
                    gr.Markdown("## Model Download")
                    
                    # Model info
                    gr.Markdown(get_model_info_text())
                    
                    gr.Markdown("---")
                    
                    # Download button and output
                    download_btn = gr.Button(
                        "📥 Download All Models (~22.5 GB)", 
                        variant="primary",
                        size="lg"
                    )
                    
                    download_output = gr.Markdown(
                        value="Click the button above to start downloading models.",
                        label="Download Progress"
                    )
                    
                    # Wire up download button
                    download_btn.click(
                        fn=run_model_download,
                        outputs=download_output,
                    ).then(
                        fn=refresh_system_status,
                        outputs=system_status_md
                    )
            
            # Update readiness banner when tab loads
            demo.load(fn=get_readiness_banner, outputs=readiness_banner)

    # --- Wiring ---

    generate_btn.click(
        fn=generate_music,
        inputs=[
            lyrics_input,
            tags_input,
            cfg_slider,
            temp_slider,
            top_k_slider,
            length_dropdown,
            seed_input,
        ],
        outputs=[status_box, audio_output, download_files],
    ).then(
        # Refresh history after generation
        fn=load_history_list,
        outputs=history_list,
    )

    refresh_hist_btn.click(load_history_list, outputs=history_list)

    history_list.select(
        fn=get_job_details,
        inputs=history_list,
        outputs=[lyrics_input, tags_input, status_box, audio_output, download_files],
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True
    )
