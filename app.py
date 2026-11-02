import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Configure the page before displaying any Streamlit elements.
st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😊",
    layout="centered",
)

# Import TensorFlow after page configuration so a missing dependency
# produces a useful message instead of an unhelpful traceback.
try:
    import tensorflow as tf
except ModuleNotFoundError:
    st.error(
        "TensorFlow is not installed in this app environment. "
        "Add the requirements.txt file shown below to your GitHub repository "
        "and reboot the Streamlit app."
    )
    st.code(
        "streamlit\n"
        "tensorflow==2.16.1\n"
        "numpy==1.26.4\n"
        "Pillow\n"
        "matplotlib",
        language="text",
    )
    st.stop()

# ------------------------------------------------------------
# Facial Expression Recognition - Streamlit app
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "facial_expression.keras"
CLASSES_PATH = BASE_DIR / "classes.json"
DEFAULT_DATASET = BASE_DIR / "FER_56_per_class" / "train"

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 8
DEFAULT_EPOCHS = 30
EXPECTED_CLASSES = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]


@st.cache_resource
def load_model_cached(model_path: str, modified_time: float):
    """Load the saved Keras model once and cache it for the app session."""
    del modified_time  # Included in the cache key to refresh after model changes.
    return tf.keras.models.load_model(model_path)


def build_model(number_of_classes: int):
    """Build the CNN used for facial-expression classification."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(64, 64, 3)),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(number_of_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_saved_model():
    """Load the model and its class labels, if both files exist."""
    if not MODEL_PATH.is_file() or not CLASSES_PATH.is_file():
        return None, None

    with CLASSES_PATH.open("r", encoding="utf-8") as file:
        labels = json.load(file)

    if not isinstance(labels, list) or not labels:
        raise ValueError("classes.json must contain a non-empty JSON list.")

    loaded_model = load_model_cached(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )

    if loaded_model.output_shape[-1] != len(labels):
        raise ValueError(
            "The number of model output classes does not match classes.json. "
            "Train the model again or use the matching class-label file."
        )

    return loaded_model, labels


def train_model(dataset_folder: str, epochs: int):
    """Train the CNN from class-specific image folders."""
    dataset_path = Path(dataset_folder).expanduser()

    if not dataset_path.is_absolute():
        dataset_path = BASE_DIR / dataset_path

    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}. "
            "Check the dataset folder path and upload the training images."
        )

    class_folders = sorted(
        folder.name for folder in dataset_path.iterdir() if folder.is_dir()
    )
    if len(class_folders) < 2:
        raise ValueError(
            "The training folder must contain at least two class subfolders."
        )

    dataset = tf.keras.utils.image_dataset_from_directory(
        str(dataset_path),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42,
    )
    class_names = dataset.class_names

    if len(class_names) != 7:
        st.warning(
            f"Found {len(class_names)} classes: {class_names}. "
            "The intended project uses seven classes: "
            + ", ".join(EXPECTED_CLASSES)
        )

    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    model = build_model(len(class_names))

    progress_bar = st.progress(0)
    status = st.empty()

    class StreamlitProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            progress_bar.progress((epoch + 1) / epochs)
            accuracy = logs.get("accuracy")
            loss = logs.get("loss")
            if accuracy is not None and loss is not None:
                status.write(
                    f"Epoch {epoch + 1}/{epochs} — "
                    f"accuracy: {accuracy:.4f}, loss: {loss:.4f}"
                )
            else:
                status.write(f"Completed epoch {epoch + 1}/{epochs}")

    model.fit(
        dataset,
        epochs=epochs,
        callbacks=[
            StreamlitProgress(),
            tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=5,
                restore_best_weights=True,
            ),
        ],
        verbose=0,
    )

    # Save the model and matching labels together.
    model.save(MODEL_PATH)
    with CLASSES_PATH.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, ensure_ascii=False, indent=2)

    # Clear the cached copy so the newly trained model is loaded next time.
    load_model_cached.clear()

    progress_bar.progress(1.0)
    status.success("Training complete. Model and class labels saved.")
    return model, class_names


# ------------------------------------------------------------
# User interface
# ------------------------------------------------------------

st.title("😊 Facial Expression Recognition")
st.write(
    "Upload a face image to classify its visible facial expression. "
    "You can train the CNN using your own image folders or load a "
    "previously trained model."
)

with st.expander("Dataset folder setup"):
    st.code(
        "FER_56_per_class/\n"
        "└── train/\n"
        "    ├── angry/\n"
        "    ├── disgust/\n"
        "    ├── fear/\n"
        "    ├── happy/\n"
        "    ├── neutral/\n"
        "    ├── sad/\n"
        "    └── surprise/",
        language="text",
    )
    st.caption(
        "Place training images inside the matching class folders. "
        "Keep test images separate from the training folder."
    )

# Initialize session state and load a saved model once.
if "model" not in st.session_state:
    try:
        saved_model, saved_classes = load_saved_model()
        st.session_state.model = saved_model
        st.session_state.class_names = saved_classes
    except Exception as error:
        st.session_state.model = None
        st.session_state.class_names = None
        st.warning(f"Saved model could not be loaded: {error}")

if "class_names" not in st.session_state:
    st.session_state.class_names = None

st.subheader("1. Train or load your model")

dataset_folder = st.text_input(
    "Training dataset folder",
    value=str(DEFAULT_DATASET.relative_to(BASE_DIR)),
)
epochs = st.number_input(
    "Training epochs",
    min_value=1,
    max_value=100,
    value=DEFAULT_EPOCHS,
    step=1,
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Train CNN model", type="primary", use_container_width=True):
        try:
            with st.spinner("Training your CNN model..."):
                trained_model, trained_classes = train_model(
                    dataset_folder,
                    int(epochs),
                )
            st.session_state.model = trained_model
            st.session_state.class_names = trained_classes
            st.success("Your facial-expression model is ready.")
        except Exception as error:
            st.error(f"Training failed: {error}")

with col2:
    if st.button("Load saved model", use_container_width=True):
        try:
            loaded_model, loaded_classes = load_saved_model()
            if loaded_model is None:
                st.warning(
                    "No saved model found. Train the model first or add "
                    "facial_expression.keras and classes.json to the app folder."
                )
            else:
                st.session_state.model = loaded_model
                st.session_state.class_names = loaded_classes
                st.success("Saved model loaded successfully.")
        except Exception as error:
            st.error(f"Could not load model: {error}")

if st.session_state.model is not None:
    st.caption(
        "Model is ready. Classes: "
        + ", ".join(st.session_state.class_names)
    )
else:
    st.info(
        "Train the model or add facial_expression.keras and classes.json "
        "to the repository root."
    )

st.divider()
st.subheader("2. Upload an image and predict")

uploaded_file = st.file_uploader(
    "Choose a face image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Predict expression", use_container_width=True):
            if st.session_state.model is None:
                st.warning("Please train or load a model first.")
            else:
                resized_image = image.resize(IMAGE_SIZE)
                image_array = np.asarray(resized_image, dtype=np.float32)
                image_batch = np.expand_dims(image_array, axis=0)

                probabilities = st.session_state.model.predict(
                    image_batch,
                    verbose=0,
                )[0]

                labels = st.session_state.class_names
                if len(probabilities) != len(labels):
                    st.error(
                        "The model output does not match the saved class labels. "
                        "Please train the model again."
                    )
                else:
                    predicted_index = int(np.argmax(probabilities))
                    predicted_label = labels[predicted_index]
                    confidence = float(probabilities[predicted_index]) * 100

                    st.success(
                        f"Predicted expression: **{predicted_label.title()}**"
                    )
                    st.metric("Model confidence", f"{confidence:.2f}%")

                    st.write("### Class probabilities")
                    sorted_indices = np.argsort(probabilities)[::-1]
                    for index in sorted_indices:
                        label = labels[int(index)]
                        score = float(probabilities[index])
                        st.write(f"{label.title()} — {score * 100:.2f}%")
                        st.progress(float(np.clip(score, 0.0, 1.0)))

    except Exception as error:
        st.error(f"Could not process this image: {error}")

st.divider()
st.caption(
    "Educational image-classification demo only. Facial-expression predictions "
    "can be inaccurate and do not establish a person's actual feelings or "
    "mental state. Use images only with appropriate permission."
)
