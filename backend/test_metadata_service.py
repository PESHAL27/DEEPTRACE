import unittest
import os
import shutil
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from app.services.metadata_service import analyze_metadata, analyze_image_metadata, analyze_video_metadata
from app.models.schemas import MetadataAnalysisResult

TEST_DIR = Path(__file__).resolve().parent / "test_scratch_metadata"


class TestMetadataService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Plain image with no EXIF
        cls.plain_image_path = TEST_DIR / "plain_no_exif.jpg"
        plain_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(cls.plain_image_path), plain_img)

        # 2. Image with synthetic EXIF tags
        cls.exif_image_path = TEST_DIR / "image_with_exif.jpg"
        pil_img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        exif = pil_img.getexif()
        exif[0x010F] = "Nikon"  # Make
        exif[0x0110] = "D850"   # Model
        exif[0x0131] = "Adobe Photoshop 2024"  # Software
        exif[0x0132] = "2024:05:12 14:30:00"  # DateTime
        pil_img.save(cls.exif_image_path, exif=exif)

        # 3. Test video
        cls.video_path = Path(__file__).resolve().parent / "test_fixtures" / "test_fake_video.mp4"
        if not cls.video_path.exists():
            cls.video_path = Path(__file__).resolve().parent.parent / "public" / "assets" / "test_fake_video.mp4"

    @classmethod
    def tearDownClass(cls):
        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR, ignore_errors=True)

    def test_image_metadata_present_with_exif(self):
        result = analyze_metadata(self.exif_image_path, "image")
        self.assertIsInstance(result, MetadataAnalysisResult)
        self.assertTrue(result.available)
        self.assertTrue(result.metadata_present)
        self.assertEqual(result.camera_make, "Nikon")
        self.assertEqual(result.camera_model, "D850")
        self.assertEqual(result.software, "Adobe Photoshop 2024")
        self.assertTrue(result.editing_software_detected)
        self.assertGreaterEqual(result.metadata_fields_count, 4)

        # Verify editing software finding
        edit_findings = [f for f in result.findings if f.type == "editing_software"]
        self.assertGreaterEqual(len(edit_findings), 1)
        self.assertEqual(edit_findings[0].severity, "warning")

    def test_image_metadata_absent_handling(self):
        result = analyze_metadata(self.plain_image_path, "image")
        self.assertIsInstance(result, MetadataAnalysisResult)
        self.assertTrue(result.available)
        self.assertFalse(result.metadata_present)
        self.assertIsNone(result.camera_make)
        self.assertIsNone(result.camera_model)
        self.assertFalse(result.editing_software_detected)

        # Verify factual absence message
        presence_findings = [f for f in result.findings if f.type == "exif_presence"]
        self.assertGreaterEqual(len(presence_findings), 1)
        self.assertIn("EXIF metadata was not present", presence_findings[0].message)
        self.assertEqual(presence_findings[0].severity, "info")

    def test_nonexistent_file_handling(self):
        nonexistent = TEST_DIR / "does_not_exist.jpg"
        result = analyze_metadata(nonexistent, "image")
        self.assertFalse(result.available)
        self.assertFalse(result.metadata_present)
        self.assertGreaterEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].type, "file_not_found")

    def test_video_container_metadata_extraction(self):
        if not self.video_path.exists():
            self.skipTest("test_fake_video.mp4 not found for video test.")

        result = analyze_metadata(self.video_path, "video")
        self.assertIsInstance(result, MetadataAnalysisResult)
        self.assertTrue(result.available)
        self.assertTrue(result.metadata_present)
        self.assertIn("codec_fourcc", result.raw_metadata)
        self.assertIn("fps", result.raw_metadata)
        self.assertIn("duration_seconds", result.raw_metadata)
        self.assertGreater(result.raw_metadata["frame_count"], 0)

        stream_findings = [f for f in result.findings if f.type in ("container_format", "temporal_stream")]
        self.assertGreaterEqual(len(stream_findings), 2)

    def test_no_fabricated_probability_or_score(self):
        # Metadata analysis must NOT contain any "probability" or "fake score"
        result = analyze_metadata(self.exif_image_path, "image")
        res_dict = result.model_dump()
        self.assertNotIn("probability", res_dict)
        self.assertNotIn("fake_probability", res_dict)
        self.assertNotIn("score", res_dict)


if __name__ == "__main__":
    unittest.main()
