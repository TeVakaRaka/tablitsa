import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

from ..models.cell import OCRAlternative

# Try to import OCR engines
TESSERACT_AVAILABLE = False
EASYOCR_AVAILABLE = False
PADDLEOCR_AVAILABLE = False

try:
    import pytesseract
    from pytesseract import Output
    # Import setup function to configure bundled Tesseract
    from .ocr_engine import setup_bundled_tesseract
    setup_bundled_tesseract()
    TESSERACT_AVAILABLE = True
except ImportError:
    pass

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    # Catch all exceptions including OSError for DLL loading failures
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except Exception:
    # Catch all exceptions including import errors
    PADDLEOCR_AVAILABLE = False

# OCR engine weights for voting (higher = more trusted)
OCR_WEIGHTS = {
    "paddleocr": 1.5,  # Primary engine - best for Russian
    "easyocr": 1.2,    # Secondary engine - neural network based
    "tesseract": 1.0,  # Tertiary engine - classic, fast
}


@dataclass
class MultiOCRResult:
    """Result from multi-OCR with voting."""
    text: str
    confidence: float
    alternatives: List[OCRAlternative]
    needs_review: bool
    agreement_score: float  # How much OCR engines agreed (0-1)


class MultiOCREngine:
    """Multi-OCR engine that combines results from multiple OCR systems."""

    def __init__(
        self,
        languages: List[str] = None,
        use_tesseract: bool = True,
        use_easyocr: bool = True,
        use_paddleocr: bool = True,
    ):
        self.languages = languages or ["ru", "en"]
        self.use_tesseract = use_tesseract and TESSERACT_AVAILABLE
        self.use_easyocr = use_easyocr and EASYOCR_AVAILABLE
        self.use_paddleocr = use_paddleocr and PADDLEOCR_AVAILABLE

        # Initialize OCR readers (lazy loading)
        self._easyocr_reader = None
        self._paddleocr_reader = None

        # Tesseract language mapping
        self._tesseract_langs = self._map_languages_tesseract(self.languages)

        # PaddleOCR language mapping
        self._paddleocr_lang = self._map_language_paddleocr(self.languages)

    def _map_languages_tesseract(self, langs: List[str]) -> str:
        """Map language codes to Tesseract format."""
        mapping = {
            "ru": "rus",
            "en": "eng",
            "de": "deu",
            "fr": "fra",
            "es": "spa",
            "it": "ita",
            "zh": "chi_sim",
            "ja": "jpn",
            "ko": "kor",
        }
        tess_langs = [mapping.get(lang, lang) for lang in langs]
        return "+".join(tess_langs)

    def _map_language_paddleocr(self, langs: List[str]) -> str:
        """Map language codes to PaddleOCR format."""
        # PaddleOCR uses specific language codes
        # Priority: ru > en for Russian documents
        mapping = {
            "ru": "ru",
            "en": "en",
            "de": "german",
            "fr": "fr",
            "es": "es",
            "it": "it",
            "zh": "ch",
            "ja": "japan",
            "ko": "korean",
        }
        # PaddleOCR only supports one language at a time
        # Prefer Russian for our use case
        for lang in langs:
            if lang in mapping:
                return mapping[lang]
        return "en"

    def _get_easyocr_reader(self):
        """Lazy load EasyOCR reader."""
        if self._easyocr_reader is None and EASYOCR_AVAILABLE:
            self._easyocr_reader = easyocr.Reader(
                self.languages,
                gpu=False,  # CPU mode for compatibility
                verbose=False
            )
        return self._easyocr_reader

    def _get_paddleocr_reader(self):
        """Lazy load PaddleOCR reader."""
        if self._paddleocr_reader is None and PADDLEOCR_AVAILABLE:
            self._paddleocr_reader = PaddleOCR(
                use_angle_cls=True,
                lang=self._paddleocr_lang,
                use_gpu=False,
                show_log=False
            )
        return self._paddleocr_reader

    def recognize(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> MultiOCRResult:
        """Recognize text using multiple OCR engines and vote."""
        # Extract region if bbox provided
        if bbox:
            x, y, w, h = bbox
            # Add padding
            pad = 3
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(image.shape[1] - x, w + 2 * pad)
            h = min(image.shape[0] - y, h + 2 * pad)
            roi = image[y:y+h, x:x+w]
        else:
            roi = image

        if roi.size == 0:
            return MultiOCRResult(
                text="",
                confidence=0.0,
                alternatives=[],
                needs_review=False,
                agreement_score=1.0
            )

        # Preprocess
        processed = self._preprocess(roi)

        # Collect results from all engines
        # Order by priority: PaddleOCR (best for Russian) > EasyOCR > Tesseract
        results: List[OCRAlternative] = []

        if self.use_paddleocr:
            paddle_result = self._paddleocr_recognize(processed)
            if paddle_result:
                results.append(paddle_result)

        if self.use_easyocr:
            easy_result = self._easyocr_recognize(processed)
            if easy_result:
                results.append(easy_result)

        if self.use_tesseract:
            tess_result = self._tesseract_recognize(processed)
            if tess_result:
                results.append(tess_result)

        # Vote and select best result
        return self._vote(results)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for OCR."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize if too small
        h, w = gray.shape[:2]
        if h < 32:
            scale = 32 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, h=10)

        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Ensure text is dark on light
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        return binary

    def _tesseract_recognize(self, image: np.ndarray) -> Optional[OCRAlternative]:
        """Recognize with Tesseract."""
        if not TESSERACT_AVAILABLE:
            return None

        try:
            # PSM 7 = single line, PSM 6 = uniform block
            h = image.shape[0]
            psm = 7 if h < 50 else 6

            data = pytesseract.image_to_data(
                image,
                lang=self._tesseract_langs,
                config=f"--psm {psm}",
                output_type=Output.DICT
            )

            texts = []
            confidences = []

            for i, conf in enumerate(data["conf"]):
                if conf > 0:
                    text = data["text"][i].strip()
                    if text:
                        texts.append(text)
                        confidences.append(conf)

            if texts:
                combined = " ".join(texts)
                avg_conf = sum(confidences) / len(confidences) / 100.0
                return OCRAlternative(
                    text=combined,
                    confidence=avg_conf,
                    source="tesseract"
                )

        except Exception as e:
            print(f"Tesseract error: {e}")

        return None

    def _easyocr_recognize(self, image: np.ndarray) -> Optional[OCRAlternative]:
        """Recognize with EasyOCR."""
        if not EASYOCR_AVAILABLE:
            return None

        try:
            reader = self._get_easyocr_reader()
            if reader is None:
                return None

            results = reader.readtext(image, detail=1, paragraph=False)

            if results:
                texts = []
                confidences = []

                for (bbox, text, conf) in results:
                    if text.strip():
                        texts.append(text.strip())
                        confidences.append(conf)

                if texts:
                    combined = " ".join(texts)
                    avg_conf = sum(confidences) / len(confidences)
                    return OCRAlternative(
                        text=combined,
                        confidence=avg_conf,
                        source="easyocr"
                    )

        except Exception as e:
            print(f"EasyOCR error: {e}")

        return None

    def _paddleocr_recognize(self, image: np.ndarray) -> Optional[OCRAlternative]:
        """Recognize with PaddleOCR (primary engine, best for Russian)."""
        if not PADDLEOCR_AVAILABLE:
            return None

        try:
            reader = self._get_paddleocr_reader()
            if reader is None:
                return None

            # PaddleOCR expects BGR or grayscale image
            results = reader.ocr(image, cls=True)

            if results and results[0]:
                texts = []
                confidences = []

                for line in results[0]:
                    if line and len(line) >= 2:
                        # Each line is [bbox, (text, confidence)]
                        text_conf = line[1]
                        if isinstance(text_conf, tuple) and len(text_conf) >= 2:
                            text = text_conf[0].strip()
                            conf = text_conf[1]
                            if text:
                                texts.append(text)
                                confidences.append(conf)

                if texts:
                    combined = " ".join(texts)
                    avg_conf = sum(confidences) / len(confidences)
                    return OCRAlternative(
                        text=combined,
                        confidence=avg_conf,
                        source="paddleocr"
                    )

        except Exception as e:
            print(f"PaddleOCR error: {e}")

        return None

    def _vote(self, results: List[OCRAlternative]) -> MultiOCRResult:
        """Vote on OCR results and select the best one using weighted voting."""
        if not results:
            return MultiOCRResult(
                text="",
                confidence=0.0,
                alternatives=[],
                needs_review=False,
                agreement_score=1.0
            )

        if len(results) == 1:
            r = results[0]
            return MultiOCRResult(
                text=r.text,
                confidence=r.confidence,
                alternatives=results,
                needs_review=r.confidence < 0.6,
                agreement_score=1.0
            )

        # Normalize texts for comparison
        normalized = [self._normalize_text(r.text) for r in results]

        # Check if all results agree
        if len(set(normalized)) == 1:
            # Full agreement - pick by weighted confidence
            best = max(results, key=lambda r: r.confidence * OCR_WEIGHTS.get(r.source, 1.0))
            return MultiOCRResult(
                text=best.text,
                confidence=best.confidence,
                alternatives=results,
                needs_review=False,
                agreement_score=1.0
            )

        # Calculate similarity between results
        similarity = self._calculate_similarity(normalized)

        if similarity > 0.8:
            # High similarity - pick by weighted confidence
            best = max(results, key=lambda r: r.confidence * OCR_WEIGHTS.get(r.source, 1.0))
            return MultiOCRResult(
                text=best.text,
                confidence=best.confidence,
                alternatives=results,
                needs_review=False,
                agreement_score=similarity
            )

        # Check for 2 out of 3 agreement
        if len(results) >= 3:
            agreement_result = self._check_majority_agreement(results, normalized)
            if agreement_result:
                return MultiOCRResult(
                    text=agreement_result.text,
                    confidence=agreement_result.confidence,
                    alternatives=results,
                    needs_review=False,
                    agreement_score=0.67  # 2/3 agreement
                )

        # Low agreement - use primary OCR (PaddleOCR) with needs_review flag
        # Sort by weight to get primary engine result
        primary = max(results, key=lambda r: OCR_WEIGHTS.get(r.source, 1.0))

        return MultiOCRResult(
            text=primary.text,
            confidence=primary.confidence,
            alternatives=results,
            needs_review=True,
            agreement_score=similarity
        )

    def _check_majority_agreement(
        self,
        results: List[OCRAlternative],
        normalized: List[str]
    ) -> Optional[OCRAlternative]:
        """Check if 2 out of 3 (or more) engines agree."""
        # Group results by normalized text
        text_groups: Dict[str, List[OCRAlternative]] = {}
        for r, norm in zip(results, normalized):
            if norm not in text_groups:
                text_groups[norm] = []
            text_groups[norm].append(r)

        # Find groups with 2+ agreements
        for norm_text, group in text_groups.items():
            if len(group) >= 2:
                # Return highest weighted result from agreeing group
                best = max(group, key=lambda r: r.confidence * OCR_WEIGHTS.get(r.source, 1.0))
                return best

        return None

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra spaces, lowercase
        return " ".join(text.lower().split())

    def _calculate_similarity(self, texts: List[str]) -> float:
        """Calculate average similarity between texts."""
        if len(texts) < 2:
            return 1.0

        try:
            from Levenshtein import ratio
            similarities = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    similarities.append(ratio(texts[i], texts[j]))
            return sum(similarities) / len(similarities) if similarities else 1.0
        except ImportError:
            # Fallback: simple character match
            if texts[0] == texts[1]:
                return 1.0
            common = sum(c1 == c2 for c1, c2 in zip(texts[0], texts[1]))
            max_len = max(len(texts[0]), len(texts[1]))
            return common / max_len if max_len > 0 else 1.0

    def _word_voting(self, results: List[OCRAlternative]) -> str:
        """Vote on individual words to build consensus."""
        # Split all results into words
        all_words = []
        for r in results:
            words = r.text.split()
            all_words.append(words)

        if not all_words:
            return ""

        # Find maximum word count
        max_words = max(len(words) for words in all_words)

        # Vote on each word position
        consensus = []
        for i in range(max_words):
            candidates = []
            for words in all_words:
                if i < len(words):
                    candidates.append(words[i])

            if candidates:
                # Pick most common word
                counter = Counter(candidates)
                best_word = counter.most_common(1)[0][0]
                consensus.append(best_word)

        return " ".join(consensus)

    @staticmethod
    def get_available_engines() -> List[str]:
        """Get list of available OCR engines in priority order."""
        engines = []
        if PADDLEOCR_AVAILABLE:
            engines.append("paddleocr")
        if EASYOCR_AVAILABLE:
            engines.append("easyocr")
        if TESSERACT_AVAILABLE:
            engines.append("tesseract")
        return engines

    @staticmethod
    def get_engine_weight(engine: str) -> float:
        """Get weight for a specific OCR engine."""
        return OCR_WEIGHTS.get(engine, 1.0)
