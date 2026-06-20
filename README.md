# AI Image Classifier

A simple web app that classifies uploaded images using a pretrained MobileNetV2 model. Upload any photo, and the app returns the top 3 most likely labels for what's in the image, along with confidence scores.

Built with Streamlit for the interface and TensorFlow/Keras for the model.

## How it works

1. The app loads MobileNetV2 with weights pretrained on ImageNet (a dataset of 1.4 million images across 1000 categories).
2. When you upload an image, it's resized to 224x224 pixels and normalized to match the format MobileNetV2 expects.
3. The image is passed through the model, which outputs predictions across all 1000 ImageNet categories.
4. The top 3 predictions are decoded into human-readable labels and displayed with their confidence scores.

This is a transfer learning project: no model training happens here. The model already learned to recognize general objects from ImageNet, and this app simply loads that pretrained knowledge and applies it.

## Tech stack

- **Python**
- **Streamlit** — web app interface
- **TensorFlow / Keras** — MobileNetV2 model and prediction pipeline
- **Pillow (PIL)** — image loading, format conversion, and resizing
- **NumPy** — array manipulation for model input

## Project setup

This project uses [uv](https://github.com/astral-sh/uv) instead of pip for dependency management.

### Prerequisites

- Python 3.9+
- uv installed ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

### Installation

Clone the repository, then run:

```bash
uv init .
uv add streamlit tensorflow pillow numpy
```

### Running the app

```bash
uv run streamlit run main.py
```

The app will open in your browser, typically at `http://localhost:8501`.

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Builds the web UI (file uploader, buttons, results display) |
| `tensorflow` | Provides MobileNetV2 and the model inference engine |
| `pillow` | Opens, converts, and resizes uploaded images |
| `numpy` | Converts images into arrays the model can process |

## Usage

1. Launch the app.
2. Upload a `.jpg` or `.png` image.
3. Click **Classify Image**.
4. View the top 3 predicted labels with confidence percentages.

## What I learned building this

This project was originally based on a tutorial, then rebuilt from scratch to make sure every line was understood rather than copied. Key concepts reinforced:

- **Transfer learning** — using a pretrained model's existing knowledge instead of training from zero.
- **Image preprocessing** — why models require a fixed input size, normalized pixel values, and a batch dimension (`np.expand_dims`) even for a single image.
- **Streamlit caching** (`@st.cache_resource`) — preventing an expensive model reload on every user interaction, since Streamlit reruns the script top to bottom on each action.
- **PIL vs OpenCV** — swapped out OpenCV (only used for one resize call) in favor of Pillow, reducing dependencies without losing functionality.

## Possible improvements

- Swap MobileNetV2 for a fine-tuned model on a custom dataset (e.g. plant diseases, food types)
- Add support for batch image uploads
- Display a confidence bar chart instead of plain text
- Deploy to Streamlit Community Cloud for a live demo link

## License

MIT
