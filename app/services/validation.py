from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(eq=False)
class VirtualFitterError(Exception):
    code: str
    message: str
    recoverable: bool = False
    http_status: int = 400
    guidance: Optional[str] = None
    detected_angle: Optional[str] = None
    product_zone: Optional[str] = None
    confidence: Optional[float] = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
        }
        if self.guidance:
            payload["guidance"] = self.guidance
        if self.detected_angle:
            payload["detected_angle"] = self.detected_angle
        if self.product_zone:
            payload["product_zone"] = self.product_zone
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return payload


@dataclass
class UserGuidance:
    title: str
    message: str
    tips: list[str] = field(default_factory=list)
    required_angles: list[str] = field(default_factory=list)
    example_description: str = ""


ERROR_GUIDANCE = {
    "VF_001": UserGuidance(
        title="No vehicle detected",
        message="We couldn't find a car in your photo. This can happen when the car is too small, partially hidden, or the image is unclear.",
        tips=[
            "Make sure your car fills at least half of the photo.",
            "Use good lighting and avoid very dark or overexposed images.",
            "Keep the entire car visible in the frame.",
            "Avoid photos with multiple vehicles.",
        ],
        required_angles=["rear", "side", "three_quarter"],
        example_description="A clear photo with the car as the main subject.",
    ),
    "VF_002_front": UserGuidance(
        title="Front view not supported",
        message="Your photo shows the front of the car. To preview this part, we need to see the area where it installs.",
        tips=[
            "For spoilers and diffusers, upload a rear view photo.",
            "For side skirts, upload a side profile photo.",
            "A rear 3/4 angle also works great for many parts.",
        ],
        required_angles=["rear", "three_quarter"],
        example_description="Stand behind the car and include the rear bumper plus trunk or hatch area.",
    ),
    "VF_002_side_for_rear": UserGuidance(
        title="Different angle needed",
        message="Your photo shows the side of the car, but this product installs at the rear. We need to see that area to build the preview.",
        tips=[
            "Walk to the back of your car before taking the photo.",
            "Show the rear bumper area clearly.",
            "A rear 3/4 angle from the corner also works.",
        ],
        required_angles=["rear", "three_quarter"],
        example_description="Stand a few meters behind the car and keep the rear centered in frame.",
    ),
    "VF_002_rear_for_side": UserGuidance(
        title="Different angle needed",
        message="Your photo shows the rear of the car, but side skirts install along the side sills. We need to see the side profile.",
        tips=[
            "Stand to the side of your car.",
            "Make sure both wheels are visible.",
            "Keep the camera level with the car for a cleaner fit.",
        ],
        required_angles=["side", "three_quarter"],
        example_description="A side profile shot showing most of the car length and both wheels.",
    ),
    "VF_003": UserGuidance(
        title="Image quality issue",
        message="We had trouble analyzing your photo. This usually happens when the image is blurry, dark, low-resolution, or partly blocked.",
        tips=[
            "Use daylight or bright, even lighting.",
            "Hold your phone steady before taking the photo.",
            "Make sure the image is sharp and in focus.",
            "Use a higher-resolution photo if possible.",
        ],
        required_angles=["rear", "side", "three_quarter"],
        example_description="A clear, well-lit photo taken during the day.",
    ),
}


def get_user_guidance(
    error_code: str,
    detected_angle: Optional[str] = None,
    product_zone: Optional[str] = None,
) -> dict[str, object]:
    if error_code == "VF_002":
        if detected_angle == "front":
            guidance_key = "VF_002_front"
        elif detected_angle == "side" and product_zone == "rear":
            guidance_key = "VF_002_side_for_rear"
        elif detected_angle == "rear" and product_zone in {"side_left", "side_right"}:
            guidance_key = "VF_002_rear_for_side"
        else:
            guidance_key = "VF_002_front"
    else:
        guidance_key = error_code

    guidance = ERROR_GUIDANCE.get(guidance_key)
    if not guidance:
        return {
            "title": "Something went wrong",
            "message": "We couldn't process your photo this time. Please try again with a clearer image.",
            "tips": [
                "Try another photo with the full car visible.",
                "Use brighter lighting and keep the image in focus.",
            ],
            "required_angles": ["rear", "side"],
            "example_description": "A clear photo with the car centered in frame.",
        }

    return {
        "title": guidance.title,
        "message": guidance.message,
        "tips": guidance.tips,
        "required_angles": guidance.required_angles,
        "example_description": guidance.example_description,
    }


class NoVehicleDetected(VirtualFitterError):
    def __init__(self, message: str = "We couldn't detect a vehicle in this image.") -> None:
        super().__init__(
            code="VF_001",
            message=message,
            recoverable=True,
            guidance="Try a clearer photo with the entire car visible and minimal background clutter.",
        )


class IncompatiblePlacementAngle(VirtualFitterError):
    def __init__(
        self,
        message: str = "This photo angle doesn't work for the selected product.",
        detected_angle: Optional[str] = None,
        product_zone: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        super().__init__(
            code="VF_002",
            message=message,
            recoverable=True,
            guidance="Use a photo that matches the product zone, like a rear photo for spoilers or a side photo for skirts.",
            detected_angle=detected_angle,
            product_zone=product_zone,
            confidence=confidence,
        )


class LowConfidenceResult(VirtualFitterError):
    def __init__(self, message: str = "We found a vehicle, but the segmentation confidence is too low.", confidence: Optional[float] = None) -> None:
        super().__init__(
            code="VF_003",
            message=message,
            recoverable=True,
            guidance="Retake the photo in daylight and include the full vehicle without heavy shadows or reflections.",
            confidence=confidence,
        )


class MissingProductMetafields(VirtualFitterError):
    def __init__(self, message: str = "This product is missing Virtual Fitter setup data.") -> None:
        super().__init__(code="VF_004", message=message, recoverable=False, http_status=422)


class OverlayAssetNotFound(VirtualFitterError):
    def __init__(self, message: str = "The product overlay asset could not be found.") -> None:
        super().__init__(code="VF_005", message=message, recoverable=False, http_status=424)


class RenderTimeout(VirtualFitterError):
    def __init__(self, message: str = "Rendering took too long and timed out.") -> None:
        super().__init__(code="VF_006", message=message, recoverable=True, http_status=504)


class InvalidImage(VirtualFitterError):
    def __init__(self, message: str = "The uploaded file is not a valid image.") -> None:
        super().__init__(code="VF_007", message=message, recoverable=True, http_status=400)


def warning(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}
