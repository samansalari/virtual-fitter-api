from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


FRIENDLY_MESSAGES = {
    "no_vehicle": {
        "title": "Hmm, that's not quite a car!",
        "message": "We're looking for your lovely motor, not {detected}.",
        "suggestions": [
            "Pop outside and snap your car from the rear or side.",
            "A car park or driveway photo works brilliantly.",
            "Make sure the whole car is in frame, not just one wheel.",
        ],
        "detected_alternatives": {
            "food": "your lunch",
            "person": "a selfie",
            "animal": "your furry friend",
            "building": "some architecture",
            "text": "a document",
            "abstract": "something abstract",
            "default": "something else entirely",
        },
    },
    "people_detected": {
        "title": "Oops! We spotted a human!",
        "message": "For privacy, we need just the car in shot with no photobombers in frame.",
        "suggestions": [
            "Ask your mate to step aside for a moment.",
            "Crop out any people before uploading if you can.",
            "A clean driveway or car park snap works best.",
        ],
    },
    "low_quality": {
        "title": "Bit blurry there!",
        "message": "We need a clearer photo to work our magic.",
        "suggestions": [
            "Try again in better lighting.",
            "Hold steady and keep the whole car in view.",
            "Use the original photo rather than a screenshot or thumbnail.",
        ],
    },
    "invalid_image": {
        "title": "That file didn't come through properly",
        "message": "We couldn't read that image, so let's try another one.",
        "suggestions": [
            "Use a JPG or PNG photo.",
            "Pick a different image from your gallery.",
        ],
    },
    "nsfw": {
        "title": "That image can't be used here",
        "message": "We can only work with straightforward photos of your car.",
        "suggestions": [
            "Please upload a normal vehicle photo only.",
            "Avoid unrelated or inappropriate images.",
        ],
    },
}


def get_friendly_error(error_type: str, context: Optional[dict[str, str]] = None) -> dict[str, object]:
    context = context or {}
    template = FRIENDLY_MESSAGES.get(error_type, FRIENDLY_MESSAGES["invalid_image"])
    message = str(template["message"])
    if "{detected}" in message:
        detected_type = context.get("detected_type", "default")
        alternatives = template.get("detected_alternatives", {})
        detected_text = alternatives.get(detected_type, alternatives.get("default", "something"))
        message = message.format(detected=detected_text)

    return {
        "title": template["title"],
        "message": message,
        "suggestions": list(template.get("suggestions", [])),
    }


class RejectionReason(Enum):
    NSFW = "nsfw_content"
    HUMAN_FACE = "human_face_detected"
    NO_VEHICLE = "no_vehicle_detected"
    LOW_QUALITY = "low_quality_image"
    INVALID_FORMAT = "invalid_image_format"


@dataclass
class ModerationResult:
    is_valid: bool
    rejection_reason: Optional[RejectionReason] = None
    rejection_message: Optional[str] = None
    confidence: float = 0.0
    details: Optional[dict[str, object]] = None


class ContentModerator:
    """
    Lightweight local moderation for shopper uploads.

    This intentionally errs on the side of simple privacy and quality checks:
    - reject obvious portrait/people photos
    - reject tiny or unreadable uploads
    - reject images that do not look vehicle-like
    """

    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    async def validate(self, image_bytes: bytes) -> ModerationResult:
        image = self._decode(image_bytes)
        if image is None:
            friendly = get_friendly_error("invalid_image")
            return ModerationResult(
                is_valid=False,
                rejection_reason=RejectionReason.INVALID_FORMAT,
                rejection_message=str(friendly["message"]),
                details={"title": str(friendly["title"]), "suggestions": friendly["suggestions"]},
            )

        height, width = image.shape[:2]
        if width < 400 or height < 300:
            friendly = get_friendly_error("low_quality")
            return ModerationResult(
                is_valid=False,
                rejection_reason=RejectionReason.LOW_QUALITY,
                rejection_message=str(friendly["message"]),
                details={"width": width, "height": height, "title": str(friendly["title"]), "suggestions": friendly["suggestions"]},
            )

        face_result = self._detect_faces(image)
        if not face_result.is_valid:
            return face_result

        nsfw_result = self._detect_sensitive_content(image)
        if not nsfw_result.is_valid:
            return nsfw_result

        vehicle_result = self._detect_vehicle_presence(image)
        if not vehicle_result.is_valid:
            return vehicle_result

        return ModerationResult(
            is_valid=True,
            confidence=vehicle_result.confidence,
            details={"width": width, "height": height, **(vehicle_result.details or {})},
        )

    @staticmethod
    def _decode(image_bytes: bytes) -> Optional[np.ndarray]:
        try:
            buffer = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except Exception as exc:  # pragma: no cover - defensive decode guard
            logger.warning("Failed to decode uploaded image: %s", exc)
            return None
        return image

    def _detect_faces(self, image: np.ndarray) -> ModerationResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frontal_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        profile_faces = self.profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        total_faces = len(frontal_faces) + len(profile_faces)

        if total_faces <= 0:
            return ModerationResult(is_valid=True)

        image_area = float(image.shape[0] * image.shape[1])
        face_area = float(sum(w * h for (_, _, w, h) in frontal_faces) + sum(w * h for (_, _, w, h) in profile_faces))
        face_ratio = face_area / max(image_area, 1.0)

        if total_faces >= 1 or face_ratio > 0.03:
            friendly = get_friendly_error("people_detected")
            return ModerationResult(
                is_valid=False,
                rejection_reason=RejectionReason.HUMAN_FACE,
                rejection_message=str(friendly["message"]),
                details={
                    "faces_detected": total_faces,
                    "face_ratio": round(face_ratio, 4),
                    "title": str(friendly["title"]),
                    "suggestions": friendly["suggestions"],
                },
            )

        return ModerationResult(is_valid=True)

    def _detect_sensitive_content(self, image: np.ndarray) -> ModerationResult:
        """
        Coarse heuristic: very high skin-tone coverage without vehicle-like structure.
        This is only meant to block obviously inappropriate uploads, not act as a full NSFW classifier.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        skin_mask_hsv = cv2.inRange(hsv, (0, 30, 60), (25, 180, 255))
        skin_mask_ycrcb = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
        skin_mask = cv2.bitwise_and(skin_mask_hsv, skin_mask_ycrcb)
        skin_ratio = float(np.count_nonzero(skin_mask) / max(skin_mask.size, 1))

        aspect_ratio = image.shape[1] / max(image.shape[0], 1)
        if skin_ratio > 0.42 and aspect_ratio < 1.1:
            friendly = get_friendly_error("nsfw")
            return ModerationResult(
                is_valid=False,
                rejection_reason=RejectionReason.NSFW,
                rejection_message=str(friendly["message"]),
                details={"skin_ratio": round(skin_ratio, 4), "title": str(friendly["title"]), "suggestions": friendly["suggestions"]},
            )

        return ModerationResult(is_valid=True)

    def _detect_vehicle_presence(self, image: np.ndarray) -> ModerationResult:
        height, width = image.shape[:2]
        aspect_ratio = width / max(height, 1)
        detected_type = self._infer_non_vehicle_type(image)
        if aspect_ratio < 0.8:
            friendly = get_friendly_error("no_vehicle", {"detected_type": detected_type})
            return ModerationResult(
                is_valid=False,
                rejection_reason=RejectionReason.NO_VEHICLE,
                rejection_message=str(friendly["message"]),
                confidence=0.2,
                details={"aspect_ratio": round(aspect_ratio, 3), "detected_type": detected_type, "title": str(friendly["title"]), "suggestions": friendly["suggestions"]},
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lower_half = gray[height // 2 :, :]
        circles = cv2.HoughCircles(
            lower_half,
            cv2.HOUGH_GRADIENT,
            dp=1.1,
            minDist=max(width // 6, 30),
            param1=60,
            param2=26,
            minRadius=max(int(width * 0.03), 12),
            maxRadius=max(int(width * 0.14), 24),
        )

        wheel_score = 0.0
        if circles is not None:
            wheel_score = min(len(circles[0]) / 2.0, 1.0)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        non_background_mask = np.logical_and(saturation > 22, value > 20)
        body_score = float(np.count_nonzero(non_background_mask) / max(non_background_mask.size, 1))

        confidence = (wheel_score * 0.65) + (min(body_score / 0.45, 1.0) * 0.35)
        if confidence < 0.3:
            friendly = get_friendly_error("no_vehicle", {"detected_type": detected_type})
            return ModerationResult(
                is_valid=False,
                rejection_reason=RejectionReason.NO_VEHICLE,
                rejection_message=str(friendly["message"]),
                confidence=confidence,
                details={
                    "wheel_score": round(wheel_score, 3),
                    "body_score": round(body_score, 3),
                    "detected_type": detected_type,
                    "title": str(friendly["title"]),
                    "suggestions": friendly["suggestions"],
                },
            )

        return ModerationResult(
            is_valid=True,
            confidence=confidence,
            details={"wheel_score": round(wheel_score, 3), "body_score": round(body_score, 3)},
        )

    def _infer_non_vehicle_type(self, image: np.ndarray) -> str:
        height, width = image.shape[:2]
        aspect_ratio = width / max(height, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_ratio = float(np.count_nonzero(edges) / max(edges.size, 1))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
        brightness_std = float(np.std(hsv[:, :, 2]) / 255.0)

        if aspect_ratio < 0.82:
            return "person"
        if edge_ratio > 0.18 and mean_saturation < 0.18:
            return "text"
        if edge_ratio > 0.16 and brightness_std < 0.14:
            return "building"
        if mean_saturation > 0.45 and brightness_std > 0.2:
            return "food"
        if mean_saturation < 0.1 and edge_ratio < 0.04:
            return "abstract"
        return "default"


content_moderator = ContentModerator()


async def validate_uploaded_image(image_bytes: bytes) -> ModerationResult:
    return await content_moderator.validate(image_bytes)
