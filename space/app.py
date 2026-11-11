"""Gradio app for the cartoonify package, deployed on Hugging Face Spaces."""

import gradio as gr

from cartoonify import cartoonify

STYLES = ["classic", "sketch", "comic", "hdr"]


def cartoonify_image(image, style: str):
    if image is None:
        raise gr.Error("Upload a photo first.")
    return cartoonify(image, style=style).cartoon


demo = gr.Interface(
    fn=cartoonify_image,
    inputs=[
        gr.Image(type="numpy", label="Upload a photo"),
        gr.Radio(STYLES, value="classic", label="Style"),
    ],
    outputs=gr.Image(label="Cartoon result"),
    title="Cartoonify in Computer Vision",
    description=(
        "Turn any photo into a cartoon with OpenCV. "
        "Pick a style: classic, pencil sketch, comic, or HDR paint. "
        "[Source code](https://github.com/fari-cloud/Cartoonify-in-Computer-vision)"
    ),
    examples=[
        ["examples/input.jpg", "classic"],
        ["examples/input.jpg", "sketch"],
        ["examples/input.jpg", "comic"],
        ["examples/input.jpg", "hdr"],
    ],
)

if __name__ == "__main__":
    demo.launch()
