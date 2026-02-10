"""
Probe set generation for DNA extraction.
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import yaml
import json
import logging
from dataclasses import dataclass
from datasets import load_dataset
import re


@dataclass
class ProbeSet:
    """Container for a set of probes with metadata."""
    probes: List[str]
    set_id: str
    description: str
    domain: str
    creation_time: str
    metadata: Dict[str, Any]


class ProbeSetGenerator:
    """
    Generator for diverse probe sets to test LLM functional behavior.
    
    Creates representative inputs that cover different aspects of language
    understanding and generation capabilities.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize probe generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.logger = logging.getLogger(__name__)
        
        # Base templates for different probe types
        self.probe_templates = self._load_probe_templates()
        
    def generate_diverse_probes(
        self,
        size: int = 100,
        domains: Optional[List[str]] = None,
        include_templates: bool = True,
        include_dataset_samples: bool = True
    ) -> ProbeSet:
        """
        Generate a diverse set of probe inputs.
        
        Args:
            size: Number of probes to generate
            domains: List of domains to include (if None, includes all)
            include_templates: Whether to include template-based probes
            include_dataset_samples: Whether to include dataset samples
            
        Returns:
            ProbeSet with generated probes
        """
        self.logger.info(f"Generating diverse probe set of size {size}")
        
        # Reset random seed to ensure reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        if domains is None:
            domains = ["general", "reasoning", "creative", "factual", "conversational"]
            
        probes = []
        
        # Template-based probes
        if include_templates:
            template_probes = self._generate_template_probes(
                size // 2, domains
            )
            probes.extend(template_probes)
            
        # Dataset-based probes
        if include_dataset_samples:
            dataset_probes = self._generate_dataset_probes(
                size - len(probes), domains
            )
            probes.extend(dataset_probes)
            
        # Fill remaining with random generation if needed
        while len(probes) < size:
            additional_probes = self._generate_random_probes(
                size - len(probes), domains
            )
            probes.extend(additional_probes)
            
        # Shuffle and trim to exact size
        random.shuffle(probes)
        probes = probes[:size]
        
        probe_set = ProbeSet(
            probes=probes,
            set_id=f"diverse_{size}_{'-'.join(domains)}",
            description=f"Diverse probe set with {size} probes across domains: {', '.join(domains)}",
            domain="mixed",
            creation_time=str(np.datetime64('now')),
            metadata={
                "size": size,
                "domains": domains,
                "generation_method": "diverse",
                "include_templates": include_templates,
                "include_dataset_samples": include_dataset_samples,
                "random_seed": self.random_seed
            }
        )
        
        self.logger.info(f"Generated {len(probes)} probes")
        return probe_set
        
    def load_standard_probes(self, probe_set_name: str = "general") -> ProbeSet:
        """
        Load a predefined standard probe set.
        
        Args:
            probe_set_name: Name of the standard probe set
            
        Returns:
            ProbeSet with loaded probes
        """
        standard_sets = {
            "general": self._create_general_probe_set(),
            "reasoning": self._create_reasoning_probe_set(),
            "creative": self._create_creative_probe_set(),
            "factual": self._create_factual_probe_set(),
            "conversational": self._create_conversational_probe_set(),
            "coding": self._create_coding_probe_set(),
            "multilingual": self._create_multilingual_probe_set()
        }
        
        if probe_set_name not in standard_sets:
            raise ValueError(f"Unknown standard probe set: {probe_set_name}")
            
        return standard_sets[probe_set_name]
        
    def validate_probe_coverage(
        self,
        probe_set: ProbeSet,
        coverage_metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Validate that probe set covers different functional aspects.
        
        Args:
            probe_set: ProbeSet to validate
            coverage_metrics: List of metrics to compute
            
        Returns:
            Dictionary with coverage metrics
        """
        if coverage_metrics is None:
            coverage_metrics = [
                "length_diversity", "lexical_diversity", "syntactic_diversity",
                "domain_coverage", "complexity_range"
            ]
            
        results = {}
        probes = probe_set.probes
        
        if "length_diversity" in coverage_metrics:
            lengths = [len(probe.split()) for probe in probes]
            results["length_diversity"] = {
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "min": np.min(lengths),
                "max": np.max(lengths),
                "cv": np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
            }
            
        if "lexical_diversity" in coverage_metrics:
            all_words = set()
            total_words = 0
            for probe in probes:
                words = probe.lower().split()
                all_words.update(words)
                total_words += len(words)
            
            results["lexical_diversity"] = {
                "unique_words": len(all_words),
                "total_words": total_words,
                "ttr": len(all_words) / total_words if total_words > 0 else 0
            }
            
        if "syntactic_diversity" in coverage_metrics:
            results["syntactic_diversity"] = self._compute_syntactic_diversity(probes)
            
        if "domain_coverage" in coverage_metrics:
            results["domain_coverage"] = self._compute_domain_coverage(probes)
            
        if "complexity_range" in coverage_metrics:
            results["complexity_range"] = self._compute_complexity_range(probes)
            
        return results
        
    def _load_probe_templates(self) -> Dict[str, List[str]]:
        """Load probe templates for different domains."""
        return {
            "general": [
                "What is {}?",
                "How does {} work?",
                "Explain the concept of {}.",
                "Tell me about {}.",
                "Describe {}."
            ],
            "reasoning": [
                "If {}, then what would happen?",
                "What is the relationship between {} and {}?",
                "Compare {} and {}.",
                "What are the pros and cons of {}?",
                "Analyze the following: {}"
            ],
            "creative": [
                "Write a story about {}.",
                "Imagine a world where {}.",
                "Create a poem about {}.",
                "Design a {} that could {}.",
                "Invent a new {} for {}."
            ],
            "factual": [
                "What year did {} happen?",
                "Who invented {}?",
                "Where is {} located?",
                "What is the capital of {}?",
                "How many {} are there?"
            ],
            "conversational": [
                "Hello, how are you?",
                "What do you think about {}?",
                "Can you help me with {}?",
                "I'm feeling {} today.",
                "What's your opinion on {}?"
            ]
        }
        
    def _generate_template_probes(
        self,
        count: int,
        domains: List[str]
    ) -> List[str]:
        """Generate probes using templates."""
        probes = []
        
        # Common entities and concepts to fill templates
        entities = [
            "artificial intelligence", "climate change", "democracy", "education",
            "technology", "music", "art", "science", "history", "philosophy",
            "economics", "politics", "culture", "nature", "space", "medicine",
            "literature", "mathematics", "psychology", "sociology"
        ]
        
        
        for domain in domains:
            if domain not in self.probe_templates:
                continue
                
            templates = self.probe_templates[domain]
            domain_count = count // len(domains)
            
            for i in range(domain_count):
                # Use a deterministic selection based on index and seed to ensure reproducibility
                template_index = (i + hash(domain) + self.random_seed) % len(templates)
                template = templates[template_index]
                
                # Fill template with deterministic entity selection
                if "{}" in template:
                    entity_index = (i + self.random_seed) % len(entities)
                    entity = entities[entity_index]
                    if template.count("{}") == 2:
                        entity2_index = (i + self.random_seed + 1) % len(entities)
                        entity2 = entities[entity2_index]
                        probe = template.format(entity, entity2)
                    else:
                        probe = template.format(entity)
                else:
                    probe = template
                    
                probes.append(probe)
                
        return probes
        
    def _generate_dataset_probes(
        self,
        count: int,
        domains: List[str]
    ) -> List[str]:
        """Generate probes by sampling from datasets."""
        probes = []
        
        try:
            # Sample from common NLP datasets
            datasets_to_sample = []
            
            if "general" in domains or "conversational" in domains:
                datasets_to_sample.append(("squad", "train", "question"))
                
            if "reasoning" in domains:
                datasets_to_sample.append(("commonsense_qa", "train", "question"))
                
            if "creative" in domains:
                datasets_to_sample.append(("writing_prompts", "train", "prompt"))
                
            # Ensure at least one sample per dataset if count > 0
            if not datasets_to_sample:
                return []
                
            samples_per_dataset = count // len(datasets_to_sample)
            if samples_per_dataset == 0:
                samples_per_dataset = 1 if count > 0 else 0
                
            for dataset_name, split, field in datasets_to_sample:
                try:
                    # FIX 1: Remove streaming=True to download the dataset locally for reproducibility
                    dataset = load_dataset(dataset_name, split=split)
                    
                    # FIX 2: Shuffle with seed and select deterministic slice instead of streaming .take()
                    selected_samples = dataset.shuffle(seed=self.random_seed).select(range(min(samples_per_dataset, len(dataset))))
                    
                    for sample in selected_samples:
                        if field in sample:
                            probe = sample[field]
                            if isinstance(probe, str) and len(probe.strip()) > 0:
                                probes.append(probe.strip())
                                
                except Exception as e:
                    self.logger.warning(f"Failed to load dataset {dataset_name}: {e}")
                    # Add deterministic fallback probes when dataset loading fails
                    fallback_probes = self._get_fallback_probes_for_domain(dataset_name)
                    probes.extend(fallback_probes[:samples_per_dataset])
                    
        except Exception as e:
            self.logger.warning(f"Dataset sampling failed: {e}")
            
        return probes[:count]
        
    def _generate_random_probes(
        self,
        count: int,
        domains: List[str]
    ) -> List[str]:
        """Generate random probes as fallback."""
        random_probes = [
            "What is the meaning of life?",
            "How do machines learn?",
            "Why is the sky blue?",
            "What makes humans creative?",
            "How does language work?",
            "What is consciousness?",
            "How do we make decisions?",
            "What is the future of technology?",
            "How do we solve complex problems?",
            "What is the nature of reality?"
        ]
        
        
        probes = []
        for i in range(count):
            # Use deterministic selection based on index and seed
            probe_index = (i + self.random_seed) % len(random_probes)
            probe = random_probes[probe_index]
            probes.append(probe)
            
        return probes
        
    def _get_fallback_probes_for_domain(self, dataset_name: str) -> List[str]:
        """Get deterministic fallback probes for a specific dataset/domain."""
        fallback_probes = {
            "squad": [
                "What is the capital of France?",
                "Who invented the telephone?",
                "When was the United Nations founded?",
                "What is photosynthesis?",
                "How does gravity work?"
            ],
            "commonsense_qa": [
                "What happens when you heat water to 100°C?",
                "Why do birds fly south in winter?",
                "What is needed to make bread rise?",
                "How do plants get energy?",
                "What causes rain?"
            ],
            "writing_prompts": [
                "Write about a day when everything went wrong.",
                "Describe a world where colors have sounds.",
                "Tell a story about finding a mysterious key.",
                "Imagine meeting your future self.",
                "Write about a library that never closes."
            ]
        }
        return fallback_probes.get(dataset_name, [
            "What is the meaning of life?",
            "How do machines learn?",
            "Why is the sky blue?",
            "What makes humans creative?",
            "How does language work?"
        ])
        
    def _create_general_probe_set(self) -> ProbeSet:
        """Create general-purpose probe set."""
        probes = [
            "What is artificial intelligence?",
            "How does language work?",
            "Explain the concept of time.",
            "What makes humans unique?",
            "How do we learn new things?",
            "What is the purpose of education?",
            "How does technology affect society?",
            "What is the meaning of success?",
            "How do we make ethical decisions?",
            "What is the role of creativity?",
            "How does communication work?",
            "What is the nature of reality?",
            "How do we solve problems?",
            "What is the importance of diversity?",
            "How do we build relationships?",
            "What is the future of humanity?",
            "How does culture shape us?",
            "What is the value of knowledge?",
            "How do we adapt to change?",
            "What makes life meaningful?"
        ]
        
        return ProbeSet(
            probes=probes,
            set_id="standard_general",
            description="General-purpose probe set covering basic concepts",
            domain="general",
            creation_time=str(np.datetime64('now')),
            metadata={"type": "standard", "domain": "general"}
        )
        
    def _create_reasoning_probe_set(self) -> ProbeSet:
        """Create reasoning-focused probe set."""
        probes = [
            "If all birds can fly, and penguins are birds, can penguins fly?",
            "What is the relationship between cause and effect?",
            "How do we know what we know?",
            "What makes an argument logical?",
            "How do we distinguish correlation from causation?",
            "What is the difference between induction and deduction?",
            "How do we make decisions under uncertainty?",
            "What are the limits of human reasoning?",
            "How do biases affect our thinking?",
            "What is the role of intuition in problem-solving?",
            "How do we evaluate evidence?",
            "What makes a good explanation?",
            "How do we handle contradictory information?",
            "What is the nature of logical paradoxes?",
            "How do we reason about the future?"
        ]
        
        return ProbeSet(
            probes=probes,
            set_id="standard_reasoning",
            description="Reasoning and logic-focused probe set",
            domain="reasoning",
            creation_time=str(np.datetime64('now')),
            metadata={"type": "standard", "domain": "reasoning"}
        )
        
    def _create_creative_probe_set(self) -> ProbeSet:
        """Create creativity-focused probe set."""
        probes = [
            "Write a story about a time-traveling librarian.",
            "Imagine a world where gravity works backwards.",
            "Create a poem about the color of music.",
            "Design a machine that runs on dreams.",
            "Invent a new sport for aliens.",
            "Describe a painting that doesn't exist yet.",
            "Write a conversation between mathematics and art.",
            "Imagine what thoughts look like.",
            "Create a recipe for happiness.",
            "Design a city for invisible people.",
            "Write a love letter from one planet to another.",
            "Invent a language that only uses colors.",
            "Describe the sound of silence.",
            "Create a dance that tells the story of rain.",
            "Imagine what happens after the last page of a book."
        ]
        
        return ProbeSet(
            probes=probes,
            set_id="standard_creative",
            description="Creative and imaginative probe set",
            domain="creative",
            creation_time=str(np.datetime64('now')),
            metadata={"type": "standard", "domain": "creative"}
        )
        
    def _create_factual_probe_set(self) -> ProbeSet:
        """Create fact-based probe set."""
        probes = [
            "What is the capital of France?",
            "Who invented the telephone?",
            "When did World War II end?",
            "What is the chemical symbol for gold?",
            "How many continents are there?",
            "What is the speed of light?",
            "Who wrote Romeo and Juliet?",
            "What is the largest planet in our solar system?",
            "When was the internet invented?",
            "What is the tallest mountain in the world?",
            "Who painted the Mona Lisa?",
            "What is the smallest unit of matter?",
            "How many bones are in the human body?",
            "What is the currency of Japan?",
            "When did humans first land on the moon?"
        ]
        
        return ProbeSet(
            probes=probes,
            set_id="standard_factual",
            description="Factual knowledge probe set",
            domain="factual",
            creation_time=str(np.datetime64('now')),
            metadata={"type": "standard", "domain": "factual"}
        )
        
    def _create_conversational_probe_set(self) -> ProbeSet:
        """Create conversational probe set."""
        probes = [
            "Hello, how are you today?",
            "What's your favorite type of music?",
            "Can you help me plan a vacation?",
            "I'm feeling stressed about work.",
            "What do you think about climate change?",
            "Tell me a joke to cheer me up.",
            "What's the best way to learn a new skill?",
            "I'm having trouble sleeping.",
            "What's your opinion on social media?",
            "How do you stay motivated?",
            "What's the most interesting thing you've learned recently?",
            "I'm trying to eat healthier.",
            "What makes a good friend?",
            "How do you handle disagreements?",
            "What's your advice for someone starting their career?"
        ]
        
        return ProbeSet(
            probes=probes,
            set_id="standard_conversational",
            description="Natural conversation probe set",
            domain="conversational",
            creation_time=str(np.datetime64('now')),
            metadata={"type": "standard", "domain": "conversational"}
        )
        
    def _create_coding_probe_set(self) -> ProbeSet:
        """Create coding and programming probe set."""
        probes = [
            "Write a function to reverse a string.",
            "How do you implement a binary search?",
            "What is the difference between a list and a tuple in Python?",
            "Explain object-oriented programming.",
            "How do you handle exceptions in code?",
            "What is recursion and when would you use it?",
            "Write a function to find the factorial of a number.",
            "How do you optimize code performance?",
            "What are design patterns in programming?",
            "Explain the concept of Big O notation.",
            "How do you implement a stack data structure?",
            "What is the difference between synchronous and asynchronous programming?",
            "Write a function to check if a string is a palindrome.",
            "How do you debug code effectively?",
            "What are the principles of clean code?"
        ]
        
        return ProbeSet(
            probes=probes,
            set_id="standard_coding",
            description="Programming and coding probe set",
            domain="coding",
            creation_time=str(np.datetime64('now')),
            metadata={"type": "standard", "domain": "coding"}
        )
        
    def _create_multilingual_probe_set(self) -> ProbeSet:
        """Create multilingual probe set."""
        probes = [
            "Hello, how are you?",
            "Hola, ¿cómo estás?",
            "Bonjour, comment allez-vous?",
            "Guten Tag, wie geht es Ihnen?",
            "こんにちは、元気ですか？",
            "你好，你好吗？",
            "Привет, как дела?",
            "Ciao, come stai?",
            "안녕하세요, 어떻게 지내세요?",
            "مرحبا، كيف حالك؟",
            "Olá, como você está?",
            "Hej, hur mår du?",
            "Hallo, hoe gaat het?",
            "Γεια σου, πώς είσαι;",
            "नमस्ते, आप कैसे हैं?"
        ]
        
        return ProbeSet(
            probes=probes,
            set_id="standard_multilingual",
            description="Multilingual probe set",
            domain="multilingual",
            creation_time=str(np.datetime64('now')),
            metadata={"type": "standard", "domain": "multilingual"}
        )
        
    def _compute_syntactic_diversity(self, probes: List[str]) -> Dict[str, Any]:
        """Compute syntactic diversity metrics."""
        question_count = sum(1 for probe in probes if probe.strip().endswith('?'))
        statement_count = len(probes) - question_count
        
        avg_sentence_length = np.mean([len(probe.split()) for probe in probes])
        
        return {
            "question_ratio": question_count / len(probes),
            "statement_ratio": statement_count / len(probes),
            "avg_sentence_length": avg_sentence_length
        }
        
    def _compute_domain_coverage(self, probes: List[str]) -> Dict[str, Any]:
        """Compute domain coverage based on keywords."""
        domain_keywords = {
            "science": ["science", "research", "experiment", "theory", "hypothesis"],
            "technology": ["technology", "computer", "software", "AI", "machine"],
            "arts": ["art", "music", "painting", "creative", "design"],
            "social": ["society", "culture", "people", "relationship", "community"],
            "philosophy": ["meaning", "existence", "reality", "consciousness", "ethics"]
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = 0
            for probe in probes:
                probe_lower = probe.lower()
                if any(keyword in probe_lower for keyword in keywords):
                    score += 1
            domain_scores[domain] = score / len(probes)
            
        return domain_scores
        
    def _compute_complexity_range(self, probes: List[str]) -> Dict[str, Any]:
        """Compute complexity range metrics."""
        complexities = []
        
        for probe in probes:
            # Simple complexity measure based on length and punctuation
            complexity = len(probe) + probe.count(',') * 2 + probe.count(';') * 3
            complexities.append(complexity)
            
        return {
            "min_complexity": min(complexities),
            "max_complexity": max(complexities),
            "mean_complexity": np.mean(complexities),
            "complexity_range": max(complexities) - min(complexities)
        }
        
    def save_probe_set(self, probe_set: ProbeSet, filepath: Path) -> None:
        """Save probe set to file."""
        data = {
            "probes": probe_set.probes,
            "set_id": probe_set.set_id,
            "description": probe_set.description,
            "domain": probe_set.domain,
            "creation_time": probe_set.creation_time,
            "metadata": probe_set.metadata
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    def load_probe_set(self, filepath: Path) -> ProbeSet:
        """Load probe set from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return ProbeSet(**data)