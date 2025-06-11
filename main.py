from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import mediapipe as mp
import io

app = FastAPI()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    h, w, _ = img_overlay.shape
    rows, cols, _ = img.shape

    y1, y2 = max(0, y), min(rows, y + h)
    x1, x2 = max(0, x), min(cols, x + w)

    y1_overlay = max(0, -y)
    y2_overlay = min(h, rows - y)
    x1_overlay = max(0, -x)
    x2_overlay = min(w, cols - x)

    if y1 >= y2 or x1 >= x2 or y1_overlay >= y2_overlay or x1_overlay >= x2_overlay:
        return img

    cropped_overlay = img_overlay[y1_overlay:y2_overlay, x1_overlay:x2_overlay]
    cropped_alpha = alpha_mask[y1_overlay:y2_overlay, x1_overlay:x2_overlay]

    alpha_factor = cropped_alpha[..., np.newaxis] / 255.0

    img[y1:y2, x1:x2] = (img[y1:y2, x1:x2] * (1 - alpha_factor) +
                         cropped_overlay * alpha_factor).astype(np.uint8)
    return img

def process_image_bytes(person_bytes, glasses_bytes):
    person_array = np.frombuffer(person_bytes, np.uint8)
    glasses_array = np.frombuffer(glasses_bytes, np.uint8)

    person_img = cv2.imdecode(person_array, cv2.IMREAD_COLOR)
    glasses_img = cv2.imdecode(glasses_array, cv2.IMREAD_UNCHANGED)

    if person_img is None or glasses_img is None:
        raise ValueError("خطأ في تحميل الصور.")

    rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)

    if not results.multi_face_landmarks:
        raise ValueError("لم يتم اكتشاف وجه في الصورة.")

    landmarks = results.multi_face_landmarks[0].landmark
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    x_left = int(left_eye.x * person_img.shape[1])
    y_left = int(left_eye.y * person_img.shape[0])
    x_right = int(right_eye.x * person_img.shape[1])
    y_right = int(right_eye.y * person_img.shape[0])

    glasses_width = int(np.hypot(x_right - x_left, y_right - y_left) * 1.82)
    glasses_height = int(glasses_width * (glasses_img.shape[0] / glasses_img.shape[1]))

    center_x = int((x_left + x_right) / 2 - glasses_width / 2) + 2
    center_y = int((y_left + y_right) / 2 - glasses_height / 2) - 4

    angle = np.arctan2(y_right - y_left, x_right - x_left)
    angle_degrees = -np.degrees(angle)

    resized_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)

    center_of_glasses = (glasses_width // 2, glasses_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center_of_glasses, angle_degrees, 1.0)
    rotated_glasses = cv2.warpAffine(
        resized_glasses,
        rotation_matrix,
        (glasses_width, glasses_height),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    overlay_bgr = rotated_glasses[:, :, :3]
    overlay_alpha = rotated_glasses[:, :, 3]

    output_img = overlay_image_alpha(person_img, overlay_bgr, center_x, center_y, overlay_alpha)

    _, encoded_img = cv2.imencode(".jpg", output_img)
    return encoded_img.tobytes()

@app.post("/apply-glasses/")
async def apply_glasses(
    person_img: UploadFile = File(...),
    glasses_img: UploadFile = File(...)
):
    person_bytes = await person_img.read()
    glasses_bytes = await glasses_img.read()

    result_img_bytes = process_image_bytes(person_bytes, glasses_bytes)
    return StreamingResponse(io.BytesIO(result_img_bytes), media_type="image/jpeg")
