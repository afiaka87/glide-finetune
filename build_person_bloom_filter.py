#!/usr/bin/env python3
"""
Build a bloom filter index for LAION dataset filtering with person detection.

This script uses multiple approaches for maximum accuracy:
1. CLIP similarity for person detection (primary, most accurate)
2. NLP-based caption analysis (fallback/additional signal)
3. Metadata filtering (size, aspect ratio, NSFW, etc.)
"""

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import io

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from PIL import Image
from pybloom_live import BloomFilter
from tqdm import tqdm

# Optional but recommended NLP libraries
try:
    import spacy
    NLP_AVAILABLE = True
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Only need POS tagging
except:
    print("Warning: spaCy not available. Install with: uv add spacy && python -m spacy download en_core_web_sm")
    NLP_AVAILABLE = False

try:
    from nltk.corpus import wordnet as wn
    import nltk
    WORDNET_AVAILABLE = True
except:
    print("Warning: NLTK WordNet not available. Install with: uv add nltk && python -c 'import nltk; nltk.download(\"wordnet\")'")
    WORDNET_AVAILABLE = False


class PersonDetector:
    """Hybrid person detection using CLIP and NLP approaches."""
    
    def __init__(
        self,
        use_clip: bool = True,
        use_nlp: bool = True,
        clip_model_name: str = "ViT-B/32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        clip_batch_size: int = 256,
        clip_threshold: float = 0.22,  # Tuned for person detection
    ):
        self.use_clip = use_clip
        self.use_nlp = use_nlp
        self.device = device
        self.clip_batch_size = clip_batch_size
        self.clip_threshold = clip_threshold
        
        # Initialize CLIP if requested
        if self.use_clip:
            try:
                import clip
                self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)
                self.clip_model.eval()
                
                # Pre-compute person-related text embeddings
                self.person_prompts = [
                    "a photo of a person",
                    "a photo of people", 
                    "a portrait of a person",
                    "a photo of a man",
                    "a photo of a woman",
                    "a photo of a child",
                    "a photo of humans",
                    "a person in the image",
                    "people in the image",
                    "a human face",
                ]
                
                with torch.no_grad():
                    text_tokens = clip.tokenize(self.person_prompts).to(device)
                    self.person_embeddings = self.clip_model.encode_text(text_tokens)
                    self.person_embeddings = F.normalize(self.person_embeddings, dim=1)
                    
                print(f"CLIP model loaded on {device}")
            except ImportError:
                print("Warning: CLIP not available. Install with: uv add clip-by-openai")
                self.use_clip = False
        
        # Initialize NLP-based detection
        if self.use_nlp:
            self._init_nlp_patterns()
    
    def _init_nlp_patterns(self):
        """Initialize NLP patterns and word lists for person detection."""
        
        # Core person words
        self.core_person_words = {
            'person', 'people', 'man', 'woman', 'men', 'women', 'boy', 'girl',
            'child', 'children', 'baby', 'babies', 'infant', 'toddler', 'teen',
            'teenager', 'adult', 'human', 'humans', 'individual', 'someone',
            'somebody', 'everyone', 'everybody', 'anyone', 'anybody', 'guy',
            'lady', 'gentleman', 'folk', 'folks'
        }
        
        # Build extended person word list from WordNet if available
        self.extended_person_words = self.core_person_words.copy()
        if WORDNET_AVAILABLE:
            self._build_wordnet_person_words()
        
        # Common occupations and roles
        self.occupations = {
            'doctor', 'teacher', 'engineer', 'nurse', 'lawyer', 'artist', 'writer',
            'musician', 'athlete', 'chef', 'pilot', 'driver', 'farmer', 'student',
            'professor', 'scientist', 'programmer', 'designer', 'photographer',
            'painter', 'dancer', 'singer', 'actor', 'actress', 'model', 'astronaut',
            'worker', 'employee', 'manager', 'director', 'president', 'officer',
            'soldier', 'police', 'firefighter', 'paramedic', 'therapist', 'counselor',
            'barber', 'stylist', 'mechanic', 'electrician', 'plumber', 'carpenter',
            'architect', 'accountant', 'banker', 'broker', 'trader', 'analyst',
            'journalist', 'reporter', 'editor', 'publisher', 'librarian', 'curator',
            'guard', 'captain', 'sailor', 'fisherman', 'hunter', 'rancher',
        }
        
        # Family relations
        self.family_words = {
            'mother', 'father', 'parent', 'daughter', 'son', 'sister', 'brother',
            'grandmother', 'grandfather', 'grandparent', 'grandchild', 'grandson',
            'granddaughter', 'aunt', 'uncle', 'cousin', 'nephew', 'niece',
            'husband', 'wife', 'spouse', 'partner', 'family', 'mom', 'dad',
            'mama', 'papa', 'mommy', 'daddy', 'sibling'
        }
        
        # Combine all person words
        self.all_person_words = (
            self.extended_person_words | 
            self.occupations | 
            self.family_words
        )
        
        # Regex patterns for person detection
        self.person_patterns = [
            # Personal pronouns (careful with false positives)
            r'\b(he|she|they|him|her|them|his|hers|their|himself|herself|themselves)\b',
            # Possessive patterns with person words
            r"\b(person|people|man|woman|child|baby|boy|girl)'s\b",
            # Common person descriptors
            r'\b(young|old|elderly|middle-aged)\s+(man|woman|person|people|boy|girl)\b',
            # Portrait/selfie indicators
            r'\b(portrait|selfie|headshot|mugshot)\b',
            # Face/body part references (often indicate people)
            r'\b(face|faces|smile|smiling|standing|sitting|walking|running)\b',
            # Group indicators
            r'\b(crowd|group|team|couple|pair|trio)\s+(of\s+)?(people|persons)?\b',
        ]
        
        # Patterns that suggest NOT a person (statues, drawings, etc.)
        self.exclude_patterns = [
            r'\b(statue|sculpture|mannequin|dummy|doll|figurine)\b',
            r'\b(cartoon|anime|drawing|sketch|painting|illustration)\s+(of\s+)?(character|person)?\b',
            r'\b(video\s+game|game)\s+character\b',
            r'\baction\s+figure\b',
            r'\b(no|without)\s+(people|person|humans?)\b',
            r'\bempty\b.*\b(street|road|room|building)\b',
        ]
    
    def _build_wordnet_person_words(self):
        """Build comprehensive person word list from WordNet."""
        if not WORDNET_AVAILABLE:
            return
        
        # Get hyponyms (subtypes) of person
        person_synsets = wn.synsets('person', pos=wn.NOUN)
        for synset in person_synsets[:3]:  # Top 3 senses of 'person'
            for hyponym in list(synset.closure(lambda s: s.hyponyms()))[:500]:  # Limit for performance
                for lemma in hyponym.lemmas():
                    word = lemma.name().lower().replace('_', ' ')
                    if len(word) < 20:  # Avoid overly specific terms
                        self.extended_person_words.add(word)
    
    def detect_person_in_caption(self, caption: str) -> Tuple[bool, float, str]:
        """
        Detect if caption likely refers to a person.
        
        Returns:
            (is_person, confidence, method_used)
        """
        caption_lower = caption.lower()
        
        # Check exclusion patterns first
        for pattern in self.exclude_patterns:
            if re.search(pattern, caption_lower):
                return False, 0.0, "excluded_pattern"
        
        # Quick check for exact word matches
        words = set(caption_lower.split())
        word_overlap = words & self.all_person_words
        if word_overlap:
            # High confidence if core person word
            if words & self.core_person_words:
                return True, 0.9, f"core_word:{list(word_overlap)[0]}"
            # Medium confidence for occupation/family
            else:
                return True, 0.7, f"extended_word:{list(word_overlap)[0]}"
        
        # Check regex patterns
        for pattern in self.person_patterns:
            if re.search(pattern, caption_lower):
                return True, 0.6, "pattern_match"
        
        # Use spaCy for POS tagging if available
        if NLP_AVAILABLE and spacy:
            doc = nlp(caption_lower)
            # Look for person-indicating noun phrases
            for token in doc:
                if token.pos_ == "NOUN":
                    # Check if it might be a person-related noun
                    if any(person_word in token.text for person_word in ['er', 'ist', 'or', 'ian']):
                        if len(token.text) > 4:  # Avoid short words
                            return True, 0.5, f"pos_pattern:{token.text}"
        
        return False, 0.0, "no_match"
    
    def detect_person_with_clip_batch(
        self, 
        images: List[Image.Image]
    ) -> List[Tuple[bool, float]]:
        """
        Detect persons in a batch of images using CLIP.
        
        Returns:
            List of (has_person, max_similarity) tuples
        """
        if not self.use_clip or not images:
            return [(False, 0.0) for _ in images]
        
        results = []
        
        # Process in batches
        for i in range(0, len(images), self.clip_batch_size):
            batch = images[i:i + self.clip_batch_size]
            
            # Preprocess images
            image_tensors = torch.stack([
                self.clip_preprocess(img) for img in batch
            ]).to(self.device)
            
            # Get image embeddings
            with torch.no_grad():
                image_embeddings = self.clip_model.encode_image(image_tensors)
                image_embeddings = F.normalize(image_embeddings, dim=1)
                
                # Compute similarities with person prompts
                similarities = image_embeddings @ self.person_embeddings.T
                max_similarities = similarities.max(dim=1)[0]
                
                # Check against threshold
                for sim in max_similarities:
                    sim_value = sim.item()
                    has_person = sim_value >= self.clip_threshold
                    results.append((has_person, sim_value))
        
        return results


def build_bloom_filter(
    tar_files: List[str],
    output_path: str = "laion_person_bloom.pkl",
    stats_path: str = "laion_filter_stats.json",
    use_clip: bool = True,
    use_nlp: bool = True,
    clip_batch_size: int = 256,
    sample_rate: float = 1.0,  # Sample rate for testing
    max_samples: Optional[int] = None,
    # Filter parameters
    min_height: int = 256,
    min_width: int = 256,
    min_similarity: float = 0.3,
    max_similarity: float = 0.95,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
    filter_nsfw: bool = True,
):
    """Build bloom filter with comprehensive person detection."""
    
    print("Initializing person detector...")
    detector = PersonDetector(
        use_clip=use_clip,
        use_nlp=use_nlp,
        clip_batch_size=clip_batch_size
    )
    
    # Initialize bloom filter (size for ~50M samples with 0.1% false positive)
    estimated_samples = 50_000_000
    bloom = BloomFilter(capacity=estimated_samples, error_rate=0.001)
    
    # Statistics tracking
    stats = {
        'total': 0,
        'accepted': 0,
        'rejected_reasons': Counter(),
        'person_detection_methods': Counter(),
        'clip_similarities': [],
        'sample_captions': {'accepted': [], 'rejected': []},
    }
    
    # Process tar files
    print(f"Processing {len(tar_files)} tar files...")
    
    # Batch for CLIP processing
    image_batch = []
    image_keys = []
    image_metadata = []
    
    for tar_file in tqdm(tar_files, desc="Processing tar files"):
        try:
            dataset = wds.WebDataset(tar_file, handler=wds.handlers.warn_and_continue, shardshuffle=False)
            
            for sample in dataset:
                if max_samples and stats['total'] >= max_samples:
                    break
                
                # Sample rate for testing
                if sample_rate < 1.0 and np.random.random() > sample_rate:
                    continue
                
                stats['total'] += 1
                key = sample['__key__']
                
                # Parse metadata
                try:
                    metadata = json.loads(sample['json'])
                    caption = sample['txt'].decode('utf-8') if 'txt' in sample else ""
                except Exception as e:
                    stats['rejected_reasons']['parse_error'] += 1
                    continue
                
                # Basic metadata filters
                accept = True
                reject_reason = None
                
                # Size filter
                h, w = metadata.get('original_height', 0), metadata.get('original_width', 0)
                if h < min_height or w < min_width:
                    accept = False
                    reject_reason = 'size'
                
                # Aspect ratio filter
                if accept and h > 0:
                    ar = w / h
                    if ar < min_aspect_ratio or ar > max_aspect_ratio:
                        accept = False
                        reject_reason = 'aspect_ratio'
                
                # CLIP similarity filter
                if accept:
                    sim = metadata.get('similarity', 0)
                    if sim < min_similarity or sim > max_similarity:
                        accept = False
                        reject_reason = 'similarity'
                
                # NSFW filter
                if accept and filter_nsfw:
                    nsfw = metadata.get('NSFW', 'UNKNOWN')
                    if nsfw in ['NSFW', 'LIKELY']:
                        accept = False
                        reject_reason = 'nsfw'
                
                # Person detection - try NLP first (fast)
                person_detected = False
                person_confidence = 0.0
                detection_method = "none"
                
                if accept and use_nlp:
                    is_person, confidence, method = detector.detect_person_in_caption(caption)
                    if is_person:
                        person_detected = True
                        person_confidence = confidence
                        detection_method = f"nlp:{method}"
                
                # If NLP didn't find person with high confidence, prepare for CLIP
                if accept and use_clip and (not person_detected or person_confidence < 0.7):
                    # Load and batch images for CLIP processing
                    if 'jpg' in sample:
                        try:
                            img = Image.open(io.BytesIO(sample['jpg'])).convert('RGB')
                            # Resize for CLIP processing (save memory)
                            img.thumbnail((512, 512), Image.Resampling.BICUBIC)
                            image_batch.append(img)
                            image_keys.append(key)
                            image_metadata.append({
                                'caption': caption,
                                'nlp_detected': person_detected,
                                'nlp_confidence': person_confidence
                            })
                        except Exception as e:
                            pass
                
                # Process CLIP batch if full
                if len(image_batch) >= clip_batch_size:
                    clip_results = detector.detect_person_with_clip_batch(image_batch)
                    
                    for (img_key, img_meta, (has_person, clip_sim)) in zip(
                        image_keys, image_metadata, clip_results
                    ):
                        stats['clip_similarities'].append(clip_sim)
                        
                        # Combine NLP and CLIP signals
                        if has_person or img_meta['nlp_detected']:
                            bloom.add(img_key)
                            stats['accepted'] += 1
                            
                            if has_person:
                                stats['person_detection_methods'][f"clip:{clip_sim:.2f}"] += 1
                            else:
                                stats['person_detection_methods']['nlp_only'] += 1
                            
                            # Save sample captions
                            if len(stats['sample_captions']['accepted']) < 100:
                                stats['sample_captions']['accepted'].append(
                                    f"{img_meta['caption'][:100]} [CLIP:{clip_sim:.2f}]"
                                )
                        else:
                            stats['rejected_reasons']['no_person'] += 1
                            if len(stats['sample_captions']['rejected']) < 100:
                                stats['sample_captions']['rejected'].append(
                                    f"{img_meta['caption'][:100]} [CLIP:{clip_sim:.2f}]"
                                )
                    
                    # Clear batches
                    image_batch = []
                    image_keys = []
                    image_metadata = []
                
                # Handle non-CLIP accepts/rejects
                elif accept and person_detected and not use_clip:
                    bloom.add(key)
                    stats['accepted'] += 1
                    stats['person_detection_methods'][detection_method] += 1
                    if len(stats['sample_captions']['accepted']) < 50:
                        stats['sample_captions']['accepted'].append(caption[:100])
                elif not accept:
                    stats['rejected_reasons'][reject_reason] += 1
                
        except Exception as e:
            print(f"Error processing {tar_file}: {e}")
            continue
    
    # Process remaining CLIP batch
    if image_batch:
        clip_results = detector.detect_person_with_clip_batch(image_batch)
        for (img_key, img_meta, (has_person, clip_sim)) in zip(
            image_keys, image_metadata, clip_results
        ):
            if has_person or img_meta['nlp_detected']:
                bloom.add(img_key)
                stats['accepted'] += 1
    
    # Save bloom filter
    print(f"\nSaving bloom filter to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(bloom, f)
    
    # Calculate and display statistics
    print(f"\n{'='*60}")
    print(f"FILTERING STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples processed: {stats['total']:,}")
    print(f"Accepted (with persons): {stats['accepted']:,} ({100*stats['accepted']/max(stats['total'],1):.1f}%)")
    print(f"Bloom filter size: ~{bloom.num_bits / 8 / 1024 / 1024:.1f} MB")
    print(f"False positive rate: {bloom.error_rate:.3%}")
    
    print(f"\n{'='*60}")
    print(f"REJECTION REASONS")
    print(f"{'='*60}")
    for reason, count in stats['rejected_reasons'].most_common():
        pct = 100 * count / max(stats['total'], 1)
        print(f"{reason:20s}: {count:8,} ({pct:5.1f}%)")
    
    if stats['person_detection_methods']:
        print(f"\n{'='*60}")
        print(f"PERSON DETECTION METHODS")
        print(f"{'='*60}")
        for method, count in stats['person_detection_methods'].most_common(10):
            pct = 100 * count / max(stats['accepted'], 1)
            print(f"{method:30s}: {count:8,} ({pct:5.1f}%)")
    
    if stats['clip_similarities']:
        print(f"\n{'='*60}")
        print(f"CLIP SIMILARITY STATISTICS")
        print(f"{'='*60}")
        sims = np.array(stats['clip_similarities'])
        print(f"Mean:   {sims.mean():.3f}")
        print(f"Median: {np.median(sims):.3f}")
        print(f"Std:    {sims.std():.3f}")
        print(f"Min:    {sims.min():.3f}")
        print(f"Max:    {sims.max():.3f}")
    
    print(f"\n{'='*60}")
    print(f"SAMPLE ACCEPTED CAPTIONS")
    print(f"{'='*60}")
    for caption in stats['sample_captions']['accepted'][:5]:
        print(f"✓ {caption}")
    
    print(f"\n{'='*60}")
    print(f"SAMPLE REJECTED CAPTIONS (no person)")
    print(f"{'='*60}")
    for caption in stats['sample_captions']['rejected'][:5]:
        print(f"✗ {caption}")
    
    # Save detailed statistics
    stats_to_save = {
        'total': stats['total'],
        'accepted': stats['accepted'],
        'rejected_reasons': dict(stats['rejected_reasons']),
        'person_detection_methods': dict(stats['person_detection_methods'].most_common(20)),
        'sample_accepted': stats['sample_captions']['accepted'][:20],
        'sample_rejected': stats['sample_captions']['rejected'][:20],
    }
    
    if stats['clip_similarities']:
        stats_to_save['clip_stats'] = {
            'mean': float(np.mean(stats['clip_similarities'])),
            'median': float(np.median(stats['clip_similarities'])),
            'std': float(np.std(stats['clip_similarities'])),
            'min': float(np.min(stats['clip_similarities'])),
            'max': float(np.max(stats['clip_similarities'])),
        }
    
    with open(stats_path, 'w') as f:
        json.dump(stats_to_save, f, indent=2)
    
    print(f"\nDetailed statistics saved to {stats_path}")
    
    return bloom, stats


def main():
    parser = argparse.ArgumentParser(
        description="Build bloom filter index for LAION person-focused dataset"
    )
    parser.add_argument(
        "--tar-files",
        nargs="+",
        required=True,
        help="Paths to WebDataset tar files (supports glob patterns)"
    )
    parser.add_argument(
        "--output",
        default="laion_person_bloom.pkl",
        help="Output path for bloom filter"
    )
    parser.add_argument(
        "--stats",
        default="laion_filter_stats.json",
        help="Output path for statistics JSON"
    )
    parser.add_argument(
        "--use-clip",
        action="store_true",
        default=True,
        help="Use CLIP for person detection (recommended)"
    )
    parser.add_argument(
        "--no-clip",
        dest="use_clip",
        action="store_false",
        help="Disable CLIP (use only NLP)"
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=256,
        help="Batch size for CLIP processing"
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Sample rate for testing (1.0 = process all)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (for testing)"
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=256,
        help="Minimum image height"
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=256,
        help="Minimum image width"
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.3,
        help="Minimum CLIP text-image similarity"
    )
    parser.add_argument(
        "--max-similarity",
        type=float,
        default=0.95,
        help="Maximum CLIP text-image similarity"
    )
    
    args = parser.parse_args()
    
    # Expand glob patterns if needed
    from glob import glob
    tar_files = []
    for pattern in args.tar_files:
        tar_files.extend(glob(pattern))
    
    if not tar_files:
        print(f"No tar files found matching patterns: {args.tar_files}")
        return
    
    print(f"Found {len(tar_files)} tar files to process")
    
    build_bloom_filter(
        tar_files=tar_files,
        output_path=args.output,
        stats_path=args.stats,
        use_clip=args.use_clip,
        clip_batch_size=args.clip_batch_size,
        sample_rate=args.sample_rate,
        max_samples=args.max_samples,
        min_height=args.min_height,
        min_width=args.min_width,
        min_similarity=args.min_similarity,
        max_similarity=args.max_similarity,
    )


if __name__ == "__main__":
    main()