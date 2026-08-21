import base64
import os
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

MODEL_PATH = "./best.pt"
model = None

app = FastAPI(title="Manga Cleaner API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_yolo_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print("YOLO model loaded successfully.")
    else:
        print(f"Warning: YOLO model path '{MODEL_PATH}' not found.")


def image_to_base64(img_bgr, ext=".jpg"):
    _, buffer = cv2.imencode(ext, img_bgr)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"


def extract_bubble_text(roi_gray: np.ndarray, roi_mask: np.ndarray, dilate: int = 2) -> np.ndarray:
    if roi_mask is None or np.sum(roi_mask > 0) < 50:
        return np.zeros_like(roi_gray)

    bubble_pixels = roi_gray[roi_mask > 0]
    bg_val = float(np.percentile(bubble_pixels, 80))

    adaptive = cv2.adaptiveThreshold(
        roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    intensity_diff = (roi_gray < (bg_val - 20)).astype(np.uint8) * 255
    dark = cv2.bitwise_or(adaptive, intensity_diff)
    dark_inside = cv2.bitwise_and(dark, roi_mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_inside, connectivity=8)

    eroded_mask = cv2.erode(roi_mask, np.ones((5, 5), np.uint8), iterations=1)
    border_margin = cv2.bitwise_xor(roi_mask, eroded_mask)
    border_margin[0, :] = 255
    border_margin[-1, :] = 255
    border_margin[:, 0] = 255
    border_margin[:, -1] = 255

    bubble_w = stats[0, cv2.CC_STAT_WIDTH]
    bubble_h = stats[0, cv2.CC_STAT_HEIGHT]
    total_area = np.sum(roi_mask > 0)

    text_mask = np.zeros_like(roi_gray)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 4:
            continue

        comp_mask = (labels == i).astype(np.uint8) * 255
        touches_border = np.any(cv2.bitwise_and(comp_mask, border_margin) > 0)
        spans_bubble = (w > 0.85 * bubble_w and h > 0.6 * bubble_h) or (
            h > 0.85 * bubble_h and w > 0.6 * bubble_w
        )
        is_huge = area > 0.4 * total_area

        if touches_border or spans_bubble or is_huge:
            continue

        text_mask = cv2.bitwise_or(text_mask, comp_mask)

    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    if dilate > 0:
        kernel = np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8)
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)

    safe_zone = cv2.erode(roi_mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.bitwise_and(text_mask, safe_zone)


@app.post("/api/process")
async def process_manga_image(
    file: UploadFile = File(...),
    conf: float = Form(0.15),
    dilate: int = Form(2),
):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})

    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    full_text_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    results = model.predict(image, conf=conf, stream=False)

    if results and results[0].boxes is not None:
        res = results[0]
        masks = getattr(res, "masks", None)
        masks_data = masks.data.cpu().numpy() if masks is not None else None

        for idx, b in enumerate(res.boxes):
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            if masks_data is not None and idx < len(masks_data):
                m_resized = cv2.resize(masks_data[idx], (img_w, img_h))
                bubble_mask = (m_resized >= 0.5).astype(np.uint8) * 255
            else:
                bubble_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                bubble_mask[y1:y2, x1:x2] = 255

            roi_gray = gray[y1:y2, x1:x2]
            roi_mask = bubble_mask[y1:y2, x1:x2]

            roi_text_mask = extract_bubble_text(roi_gray, roi_mask, dilate=dilate)
            full_text_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                full_text_mask[y1:y2, x1:x2], roi_text_mask
            )

    cleaned = cv2.inpaint(image, full_text_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return JSONResponse(
        {
            "success": True,
            "original_image": image_to_base64(image),
            "cleaned_image": image_to_base64(cleaned),
        }
    )

os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
