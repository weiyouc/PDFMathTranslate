import os
import re
import subprocess
import tempfile
from pathlib import Path
from PIL import Image
import shutil

import gradio as gr
from pdf2image import convert_from_path
from pdf2zh.pdf2zh import extract_text


class AppState:
    def __init__(self):
        self.current_page = 0
        self.original_pages = None
        self.translated_pages = None
        self.original_summary = None
        self.translated_summary = None

app_state = AppState()


def upload_file(file, service, progress=gr.Progress()):
    """Handle file upload, validation, and initial preview."""
    if not file or not os.path.exists(file):
        return None, None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    progress(0.3, desc="Converting PDF for preview...")
    try:
        # Convert all pages but only show first page initially
        images = convert_from_path(file)
        preview_image = images[0] if images else None
        
        # Store all pages in a hidden state
        app_state.original_pages = images
        app_state.current_page = 0
        
        return (file, preview_image, None, 
                gr.update(visible=True),  # translate_btn
                gr.update(visible=True),  # summarize_original_btn
                gr.update(visible=False))  # summarize_translated_btn
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return None, None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def generate_summaries(input_pdf, final_output, service):
    """Generate summaries for both original and translated documents."""
    try:
        outfp, original_summary, translated_summary = extract_text(
            files=[str(input_pdf)],
            outfile=str(final_output),
            service=service,
            summarize=True
        )
        
        # Store summaries in app state
        app_state.original_summary = original_summary or "Summary generation failed."
        app_state.translated_summary = translated_summary or "Summary generation failed."
        
    except Exception as e:
        print(f"Error during translation/summarization: {e}")
        app_state.original_summary = f"Error generating summary: {str(e)}"
        app_state.translated_summary = f"Error generating summary: {str(e)}"
        raise

def translate(file_path, service="ollama:qwen2.5:7b", progress=gr.Progress()):
    """Translate PDF content using selected service."""
    if not file_path:
        return None, None, gr.update(visible=False), gr.update(visible=False)

    progress(0, desc="Starting translation...")
    lang_to = "zh"

    try:
        # Create temp directory just for the input/output files
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
            # Setup input and output paths
            input_pdf = temp_path / "input.pdf"
            final_output = Path("gradio_files") / "outputs" / f"translated_{os.path.basename(file_path)}"

            # Copy input file to temp directory
            progress(0.2, desc="Preparing files...")
            shutil.copy2(file_path, input_pdf)

            # Get the script directory for proper module imports
            script_dir = Path(__file__).parent.parent

            # Build command using the script directory as working directory
            cmd = f"cd '{script_dir}' && python -m pdf2zh.pdf2zh '{input_pdf}' -s {service}"
            
            print(f"Executing command: {cmd}")
            
            # Execute translation command
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )

            # Monitor progress from command output
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(f"Command output: {output.strip()}")
                    # Look for percentage in output
                    match = re.search(r"(\d+)%", output.strip())
                    if match:
                        percent = int(match.group(1))
                        # Map command progress (0-100%) to our progress range (30-80%)
                        progress_val = 0.3 + (percent * 0.5 / 100)
                        progress(progress_val, desc=f"Translating content: {percent}%")

            # Get the return code
            return_code = process.poll()
            print(f"Command completed with return code: {return_code}")

            # Check if translation was successful - look in script_dir instead of temp_dir
            translated_file = script_dir / f"{input_pdf.stem}-{lang_to}.pdf"
            print(f"Looking for translated file at: {translated_file}")

            if not translated_file.exists():
                print(f"Translation failed: Output file not found at {translated_file}")
                return None, None, gr.update(visible=False), gr.update(visible=False)

            # Copy the translated file to a permanent location
            progress(0.8, desc="Saving translated file...")
            with open(translated_file, "rb") as src, open(final_output, "wb") as dst:
                dst.write(src.read())

            # Generate preview of translated PDF
            progress(0.9, desc="Generating preview...")
            try:
                # Convert all pages of translated PDF
                translated_images = convert_from_path(str(final_output))
                translated_preview = translated_images[0] if translated_images else None
                
                # Store translated pages
                app_state.translated_pages = translated_images
                app_state.current_page = 0
                
            except Exception as e:
                print(f"Error generating preview: {e}")
                translated_preview = None

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"Error in translation process: {e}")
        return None, None, gr.update(visible=False), gr.update(visible=False)

    progress(1.0, desc="Translation complete!")
    return (str(final_output), translated_preview, gr.update(visible=True), 
            gr.update(visible=True))  # Make translation summary button visible


def change_page(direction):
    """Handle page navigation."""
    if not hasattr(app_state, 'original_pages') or not app_state.original_pages:
        return None, None
    
    total_pages = len(app_state.original_pages)
    app_state.current_page = (app_state.current_page + direction) % total_pages
    
    original_preview = app_state.original_pages[app_state.current_page]
    translated_preview = (app_state.translated_pages[app_state.current_page] 
                        if hasattr(app_state, 'translated_pages') and app_state.translated_pages 
                        else None)
    
    return original_preview, translated_preview


def summarize_document(file_path, is_translated=False, progress=gr.Progress()):
    """Summarize the document content using Ollama."""
    if not file_path:
        return None
    
    progress(0.3, desc="Extracting text for summary...")
    
    try:
        # Use the translated PDF if is_translated is True and it exists
        if is_translated and hasattr(app_state, 'translated_pages'):
            pages = app_state.translated_pages
        elif hasattr(app_state, 'original_pages'):
            pages = app_state.original_pages
        else:
            return "No document loaded to summarize."
        
        # Extract text from all pages
        text = ""
        for page in pages:
            # Convert PIL Image to text using pytesseract (you'll need to install this)
            text += pytesseract.image_to_string(page, lang='chi_sim' if is_translated else 'eng')
        
        progress(0.6, desc="Generating summary...")
        
        # Use Ollama for summarization
        prompt = f"Please provide a concise summary of the following text:\n\n{text}"
        
        # Call Ollama API
        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   "model": "qwen2.5:7b",
                                   "prompt": prompt,
                                   "stream": False
                               })
        
        if response.status_code == 200:
            summary = response.json()['response']
            progress(1.0, desc="Summary complete!")
            return summary
        else:
            return "Error generating summary."
            
    except Exception as e:
        print(f"Error in summarization: {e}")
        return f"Error generating summary: {str(e)}"


def show_original_summary():
    """Display the original document summary with error handling."""
    try:
        if not hasattr(app_state, 'original_summary'):
            return gr.update(value="Please translate the document first.", visible=True)
        return gr.update(value=app_state.original_summary, visible=True)
    except Exception as e:
        return gr.update(value=f"Error displaying summary: {str(e)}", visible=True)

def show_translated_summary():
    """Display the translated document summary with error handling."""
    try:
        if not hasattr(app_state, 'translated_summary'):
            return gr.update(value="Please translate the document first.", visible=True)
        return gr.update(value=app_state.translated_summary, visible=True)
    except Exception as e:
        return gr.update(value=f"Error displaying summary: {str(e)}", visible=True)


with gr.Blocks(title="PDF Translation", css="footer {display: none} .container {max-width: 100% !important; padding: 0 !important}") as app:
    gr.Markdown("# PDF Translation")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                service = gr.Dropdown(
                    label="Service",
                    choices=["ollama:qwen2.5:7b"],
                    value="ollama:qwen2.5:7b",
                    scale=1,
                    min_width=100,
                )

                file_input = gr.File(
                    label="Upload",
                    file_count="single",
                    file_types=[".pdf"],
                    type="filepath",
                    scale=1,
                )

                translate_btn = gr.Button("Translate", variant="primary", visible=False)
                
                # Add summary buttons
                with gr.Row():
                    summarize_original_btn = gr.Button("📝 Summarize Original", visible=False)
                    summarize_translated_btn = gr.Button("📝 Summarize Translation", visible=False)
                
                # Add summary output textboxes
                original_summary = gr.Textbox(label="Original Summary", visible=False, lines=5)
                translated_summary = gr.Textbox(label="Translation Summary", visible=False, lines=5)
                
                output_file = gr.File(label="Download Translation", visible=False)
                gr.Markdown("[Version 1.0]")

        with gr.Column(scale=15):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Original Document")
                    preview = gr.Image(
                        label="Preview",
                        visible=True,
                        elem_classes=["pdf-preview"],
                        height=600,
                        container=True,
                        show_download_button=False,
                    )
                with gr.Column():
                    gr.Markdown("### Translated Document")
                    translated_preview = gr.Image(
                        label="Translation Preview",
                        visible=True,
                        elem_classes=["pdf-preview"],
                        height=600,
                        container=True,
                        show_download_button=False,
                    )
            
            # Navigation controls
            with gr.Row(elem_classes=["navigation-controls"]):
                prev_btn = gr.Button("← Previous", scale=2)
                current_page = gr.Markdown("Page: 1", elem_classes=["page-number"])
                next_btn = gr.Button("Next →", scale=2)

    # Update CSS
    gr.Markdown("""
        <style>
            .pdf-preview {
                max-height: 600px !important;
                border: 1px solid #ddd !important;
            }
            .pdf-preview img {
                max-width: none;
                height: auto;
                display: block;
                margin: 0 auto;
            }
            .navigation-controls {
                justify-content: center;
                margin-top: 1rem;
                gap: 0.5rem;
            }
            .page-number {
                text-align: center;
                margin: 0;
                padding: 0.5rem;
                min-width: 100px;
            }
        </style>
    """)

    # Update event handlers
    def update_page_and_preview(direction):
        original, translated = change_page(direction)
        total_pages = len(app_state.original_pages) if hasattr(app_state, 'original_pages') else 1
        page_text = f"Page: {app_state.current_page + 1} / {total_pages}"
        return original, translated, page_text

    prev_btn.click(
        lambda: update_page_and_preview(-1),
        outputs=[preview, translated_preview, current_page],
    )
    next_btn.click(
        lambda: update_page_and_preview(1),
        outputs=[preview, translated_preview, current_page],
    )

    # Update event handlers to include translated_preview
    file_input.upload(
        upload_file,
        inputs=[file_input, service],
        outputs=[file_input, preview, translated_preview, translate_btn, 
                 summarize_original_btn, summarize_translated_btn],
    )

    translate_btn.click(
        translate,
        inputs=[file_input, service],
        outputs=[output_file, translated_preview, output_file, summarize_translated_btn],
    )

    summarize_original_btn.click(
        fn=show_original_summary,
        outputs=[original_summary],
    )

    summarize_translated_btn.click(
        fn=show_translated_summary,
        outputs=[translated_summary],
    )

app.launch(debug=True, inbrowser=True, share=True)
