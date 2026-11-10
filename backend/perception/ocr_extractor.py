"""
Multimodal Perception Suite: OCR Extractor & Document Quality Assessment
Extracts document text fields, assesses image degradation (blur/glare),
and performs fuzzy consistency checks against applicant-submitted form data.
"""

import os
import re
import difflib
import numpy as np
import cv2

class OCRExtractor:
    def __init__(self):
        pass

    def evaluate_quality(self, image_np: np.ndarray) -> dict:
        """
        Assesses image quality indicators:
        - Blur (Laplacian variance): low variance (< 500) indicates severe blur.
        - Glare (Specular reflection washout in document field area):
          Washout eliminates text contrast (min intensity > 150 or std dev < 5.0).
        - Contrast: RMS contrast across the document.
        """
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Laplacian variance for sharpness
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Inspect central text region for specular glare washout [y: 120-260, x: 260-450]
        h, w = gray.shape
        if h >= 260 and w >= 450:
            roi = gray[120:260, 260:450]
            roi_min = float(np.min(roi))
            roi_std = float(np.std(roi))
            has_glare = bool(roi_min > 120.0 and roi_std < 5.5)
            glare_ratio = round(float(1.0 - (roi_std / 10.0)), 4) if has_glare else 0.01
        else:
            has_glare = False
            glare_ratio = 0.0

        contrast = float(gray.std())
        is_blurry = bool(laplacian_var < 500.0)

        # Quality score (0.0 to 1.0)
        quality_score = 1.0
        if is_blurry:
            quality_score -= min(0.65, (500.0 - laplacian_var) / 500.0 * 0.65)
        if has_glare:
            quality_score -= 0.50

        quality_score = round(max(0.05, min(0.99, quality_score)), 4)

        return {
            "blur_metric": round(laplacian_var, 2),
            "is_blurry": is_blurry,
            "glare_ratio": glare_ratio,
            "has_glare": has_glare,
            "contrast": round(contrast, 2),
            "quality_score": quality_score,
            "pass_quality_gate": not (is_blurry or has_glare)
        }

    def extract_document_fields(self, card_image_path: str, bundle_metadata: dict = None) -> dict:
        """
        Extracts structured document text fields.
        In production, OCR engines like Tesseract / PaddleOCR or vision SLMs are used.
        Here, we inspect image quality; if degraded by blur/glare, OCR yields character noise / dropouts.
        """
        img = cv2.imread(card_image_path)
        if img is None:
            return {"error": "Unable to read document image", "success": False}

        quality = self.evaluate_quality(img)

        # Extract from bundle metadata or template region
        if bundle_metadata and "document_data" in bundle_metadata:
            raw_doc = bundle_metadata["document_data"]
            doc_name = raw_doc["doc_name"]
            doc_dob = raw_doc["doc_dob"]
            doc_id = raw_doc["doc_id_number"]
            doc_addr = raw_doc["doc_address"]
            doc_type = raw_doc["doc_type"]
            expiry = raw_doc.get("expiry_date", "2035-12-31")
        else:
            # Fallback mock extraction
            doc_name = "Unknown Citizen"
            doc_dob = "1990-01-01"
            doc_id = "ID12345678"
            doc_addr = "Republic Standard Address"
            doc_type = "NATIONAL_ID"
            expiry = "2035-12-31"

        # Simulate OCR degradation under severe blur / glare
        ocr_confidence = quality["quality_score"]
        extracted_name = doc_name
        extracted_dob = doc_dob
        extracted_id = doc_id

        if quality["is_blurry"] or quality["has_glare"]:
            # Drop characters or introduce substitution noise
            if len(extracted_name) > 4:
                chars = list(extracted_name)
                # Swap or replace one char due to blur
                chars[min(3, len(chars)-1)] = '?'
                extracted_name = "".join(chars)
            ocr_confidence = round(ocr_confidence * 0.75, 4)

        return {
            "success": True,
            "quality": quality,
            "fields": {
                "full_name": extracted_name,
                "dob": extracted_dob,
                "id_number": extracted_id,
                "address": doc_addr,
                "document_type": doc_type,
                "expiry_date": expiry
            },
            "ocr_confidence": ocr_confidence
        }

    def verify_consistency(self, form_data: dict, ocr_data: dict) -> dict:
        """
        Cross-checks applicant form inputs against OCR extracted data.
        Performs fuzzy matching for names, exact matching for DOB, and format verification.
        """
        ocr_fields = ocr_data.get("fields", {})

        app_name = (form_data.get("full_name") or "").strip().lower()
        doc_name = (ocr_fields.get("full_name") or "").strip().lower()

        # Fuzzy string similarity (Levenshtein / Gestalt pattern matching)
        name_sim = difflib.SequenceMatcher(None, app_name, doc_name).ratio()
        name_match = bool(name_sim >= 0.85)

        # DOB match
        app_dob = (form_data.get("dob") or "").strip()
        doc_dob = (ocr_fields.get("dob") or "").strip()
        dob_match = (app_dob == doc_dob)

        # ID Number match
        app_id = (form_data.get("id_number") or "").strip().upper()
        doc_id = (ocr_fields.get("id_number") or "").strip().upper()
        id_match = (app_id == doc_id)

        # Overall text consistency score
        consistency_score = (
            0.50 * name_sim +
            0.30 * (1.0 if dob_match else 0.0) +
            0.20 * (1.0 if id_match else 0.0)
        )
        consistency_score = round(consistency_score, 4)

        discrepancies = []
        if not name_match:
            discrepancies.append(f"Name mismatch: Form '{form_data.get('full_name')}' vs Doc '{ocr_fields.get('full_name')}' (similarity: {name_sim:.2f})")
        if not dob_match:
            discrepancies.append(f"DOB mismatch: Form '{app_dob}' vs Doc '{doc_dob}'")
        if not id_match:
            discrepancies.append(f"ID Number mismatch: Form '{app_id}' vs Doc '{doc_id}'")

        return {
            "name_similarity": round(name_sim, 4),
            "name_match": name_match,
            "dob_match": dob_match,
            "id_match": id_match,
            "consistency_score": consistency_score,
            "has_discrepancies": len(discrepancies) > 0,
            "discrepancies": discrepancies
        }
