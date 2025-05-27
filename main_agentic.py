#!/usr/bin/env python3
"""
Minimal Agentic Article Refinement System
-----------------------------------------
Overview: Proposer -> Reviewer -> Reviser
Quick MVP implementation with vLLM engine interface
Model: Microsoft Phi-4-mini-instruct
"""

import json
import uuid
import time
from typing import List, Optional
from dataclasses import dataclass
from vllm import LLM, SamplingParams


# ===== CONFIGURATION =====
@dataclass
class Config:
    model_name: str = "microsoft/Phi-4-mini-instruct"  # MVP with known template
    temperature: float = 0.7
    max_tokens: int = 1000
    max_iterations: int = 2
    tensor_parallel_size: int = 2
    stop_tokens: List[str] = None

    def __post_init__(self):
        # Avoid Python mutable default values - ensures each instance gets its own list
        if self.stop_tokens is None:
            self.stop_tokens = [
                "<|end|>"
            ]  # NOTE: fully specified by the model - Phi4-mini-instruct: https://huggingface.co/microsoft/Phi-4-mini-instruct#chat-format


# ===== DATA MODELS =====
@dataclass
class KnowledgeTriplet:
    subject: str
    relation: str
    effect: str


@dataclass
class Article:
    id: str
    content: str
    triplets: List[KnowledgeTriplet]
    word_count: int


@dataclass
class Evaluation:
    id: str
    article_id: str
    style_score: int
    integration_score: int
    flow_score: int
    factuality_score: int
    average_score: float
    suggestions: str
    raw_text: str


@dataclass
class PromptRevision:
    id: str
    original_prompt: str
    improved_prompt: str
    reasoning: str


# ===== PROMPTS =====
TRIPLET_PROMPT = """Generate 2-3 knowledge triplets about food insecurity causes and effects.
Return only valid JSON in this format:
{
    "triplets": [
        {"subject": "cause", "relation": "leads to", "effect": "food insecurity impact"}
    ]
}"""

ARTICLE_PROMPT = """Write a 400-word news article about food insecurity.
Include these causal relationships naturally: {triplets}

Requirements:
- Professional journalism style
- Include quotes from affected people and officials  
- Weave cause-effect relationships into narrative
- Clear, engaging story structure"""

EVALUATION_PROMPT = """Evaluate this food insecurity article:

ARTICLE:
{article}

REQUIRED RELATIONSHIPS:
{triplets}

Rate 1-10 on:
Style: [SCORE] - Professional journalism quality
Integration: [SCORE] - Natural cause-effect weaving  
Flow: [SCORE] - Story readability
Factuality: [SCORE] - Realistic claims

Then provide 2-3 specific improvement suggestions."""

REVISION_PROMPT = """Improve this prompt based on evaluation:

ORIGINAL PROMPT:
{original_prompt}

EVALUATION & SUGGESTIONS:
{evaluation}

Provide:
1. Brief reasoning for changes
2. Improved prompt that addresses the issues

Format:
REASONING: [explanation]
IMPROVED PROMPT: [new prompt]"""


# ===== MAIN SYSTEM =====
class AgenticSystem:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.llm = LLM(
            model=self.config.model_name, trust_remote_code=True, tensor_parallel_size=2
        )
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stop=self.config.stop_tokens,  # arrest generation when specified stop tokens are generated
        )
        print(f"Initialised agentic system powered by{self.config.model_name}")

    def _format_chat_prompt(self, system_msg: str, user_msg: str) -> str:
        """Format prompt using Phi-4 chat template."""
        return f"<|system|>{system_msg}<|end|><|user|>{user_msg}<|end|><|assistant|>"  # implement the Phi4 chat template

    def _generate(
        self,
        prompt: str,
        system_msg: str = "You are a professional journalist specialising in the geo-political, socio-economic, technological and environmental causes of food insecurity risks across Africa and the Middle East. You write in a professional, engaging, factual and informative style for established and trusted news outlets.",
    ) -> str:
        """Helper to generate text with chat template."""
        formatted_prompt = self._format_chat_prompt(system_msg, prompt)
        outputs = self.llm.generate([formatted_prompt], self.sampling_params)
        return outputs[0].outputs[0].text.strip()

    # ===== AGENT 1: PROPOSER =====
    def generate_triplets(self) -> List[KnowledgeTriplet]:
        """Generate knowledge triplets."""
        response = self._generate(TRIPLET_PROMPT)

        try:
            # Extract JSON from response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx >= 0 and end_idx > 0:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                return [KnowledgeTriplet(**t) for t in data["triplets"]]
        except:
            print("⚠️ Defaulting to predefined triplets")
            pass

        # Fallback
        return [
            KnowledgeTriplet("Climate change", "causes", "crop failures"),
            KnowledgeTriplet("Economic instability", "leads to", "food access issues"),
        ]

    def generate_article(
        self, prompt: str, triplets: List[KnowledgeTriplet]
    ) -> Article:
        """Generate article from prompt and triplets."""
        triplets_text = "; ".join(
            [f"{t.subject} {t.relation} {t.effect}" for t in triplets]
        )
        full_prompt = (
            ARTICLE_PROMPT.format(triplets=triplets_text)
            + f"\n\nSpecific instructions: {prompt}"
        )

        content = self._generate(full_prompt)

        return Article(
            id=str(uuid.uuid4()),
            content=content,
            triplets=triplets,
            word_count=len(content.split()),
        )

    # ===== AGENT 2: REVIEWER =====
    def evaluate_article(self, article: Article) -> Evaluation:
        """Evaluate article quality."""
        triplets_text = "; ".join(
            [f"{t.subject} {t.relation} {t.effect}" for t in article.triplets]
        )

        eval_prompt = EVALUATION_PROMPT.format(
            article=article.content[:1500],  # Truncate to fit context
            triplets=triplets_text,
        )

        response = self._generate(eval_prompt)

        # Parse scores (basic regex-free parsing)
        scores = [5, 5, 5, 5]  # defaults
        lines = response.lower().split("\n")

        for line in lines:
            if "style:" in line:
                try:
                    scores[0] = int(
                        "".join(filter(str.isdigit, line.split("style:")[1][:3]))
                    )
                except:
                    pass
            elif "integration:" in line:
                try:
                    scores[1] = int(
                        "".join(filter(str.isdigit, line.split("integration:")[1][:3]))
                    )
                except:
                    pass
            elif "flow:" in line:
                try:
                    scores[2] = int(
                        "".join(filter(str.isdigit, line.split("flow:")[1][:3]))
                    )
                except:
                    pass
            elif "factuality:" in line:
                try:
                    scores[3] = int(
                        "".join(filter(str.isdigit, line.split("factuality:")[1][:3]))
                    )
                except:
                    pass

        # Extract suggestions (everything after scores)
        suggestions = response
        if any(
            score_word in response.lower()
            for score_word in ["style:", "integration:", "flow:", "factuality:"]
        ):
            # Find last score line and take everything after
            lines = response.split("\n")
            suggestion_start = 0
            for i, line in enumerate(lines):
                if any(
                    word in line.lower()
                    for word in ["style:", "integration:", "flow:", "factuality:"]
                ):
                    suggestion_start = i + 1
            suggestions = "\n".join(lines[suggestion_start:]).strip()

        return Evaluation(
            id=str(uuid.uuid4()),
            article_id=article.id,
            style_score=scores[0],
            integration_score=scores[1],
            flow_score=scores[2],
            factuality_score=scores[3],
            average_score=sum(scores) / 4,
            suggestions=suggestions,
            raw_text=response,
        )

    # ===== AGENT 3: REVISER =====
    def revise_prompt(
        self, original_prompt: str, evaluation: Evaluation
    ) -> PromptRevision:
        """Generate improved prompt based on evaluation."""
        revision_prompt = REVISION_PROMPT.format(
            original_prompt=original_prompt, evaluation=evaluation.raw_text
        )

        response = self._generate(revision_prompt)

        # Split reasoning and improved prompt
        reasoning = response
        improved_prompt = original_prompt  # fallback

        if "IMPROVED PROMPT:" in response:
            parts = response.split("IMPROVED PROMPT:", 1)
            if len(parts) == 2:
                reasoning = parts[0].replace("REASONING:", "").strip()
                improved_prompt = parts[1].strip()
        elif "reasoning:" in response.lower() and "prompt:" in response.lower():
            # Flexible parsing
            lines = response.split("\n")
            in_prompt = False
            prompt_lines = []
            reasoning_lines = []

            for line in lines:
                if "prompt:" in line.lower():
                    in_prompt = True
                    prompt_lines.append(line.split(":", 1)[-1].strip())
                elif in_prompt:
                    prompt_lines.append(line)
                else:
                    reasoning_lines.append(line)

            if prompt_lines:
                improved_prompt = "\n".join(prompt_lines).strip()
            if reasoning_lines:
                reasoning = "\n".join(reasoning_lines).strip()

        return PromptRevision(
            id=str(uuid.uuid4()),
            original_prompt=original_prompt,
            improved_prompt=improved_prompt,
            reasoning=reasoning,
        )

    # ===== MAIN WORKFLOW =====
    def run_refinement_cycle(
        self, initial_prompt: str = "Write an engaging article about food insecurity"
    ) -> None:
        """Run complete refinement cycle."""
        start_time = time.time()
        print("=" * 60)
        print("AGENTIC ARTICLE REFINEMENT SYSTEM")
        print("=" * 60)

        current_prompt = initial_prompt

        # Step 1: Generate triplets - fixed ground truth; no need to revise iteratively
        print("🔄 Generating knowledge triplets...")
        triplets = self.generate_triplets()
        print(f"✓ Generated {len(triplets)} triplets")

        for iteration in range(1, self.config.max_iterations + 1):
            print(f"\n--- ITERATION {iteration} ---")

            # Step 2: Generate article
            print("📝 Generating article...")
            article = self.generate_article(current_prompt, triplets)
            print(f"✓ Generated article ({article.word_count} words)")

            # Step 3: Evaluate article
            print("🔍 Evaluating article...")
            evaluation = self.evaluate_article(article)
            print(
                f"✓ Evaluation complete (avg score: {evaluation.average_score:.1f}/10)"
            )

            # Display results
            print(f"\n📊 SCORES:")
            print(f"  Style: {evaluation.style_score}/10")
            print(f"  Integration: {evaluation.integration_score}/10")
            print(f"  Flow: {evaluation.flow_score}/10")
            print(f"  Factuality: {evaluation.factuality_score}/10")
            print(f"  Average: {evaluation.average_score:.1f}/10")

            print(f"\n📄 ORIGINAL SYNTHETIC ARTICLE EXCERPT:")
            print(
                " ".join(
                    article.content.split()[: int(len(article.content.split()) * 0.33)]
                )
                + "..."
            )  # print first 33% of article content

            # Step 4: Revise prompt (if not final iteration)
            if iteration < self.config.max_iterations:
                print("\n🔧 Revising prompt based on evaluation...")
                revision = self.revise_prompt(current_prompt, evaluation)
                current_prompt = revision.improved_prompt
                print("✓ Prompt revised")
                print(f"\n💡 REASONING: {revision.reasoning}")
            else:
                print(f"\n🎯 FINAL ARTICLE:\n{'-'*40}")
                print(article.content)
                print(f"{'-'*40}")

        end_time = time.time()
        runtime = end_time - start_time
        print(f"\n✅ Refinement cycle complete!")
        print(f"⏱️ Total runtime: {runtime:.2f} seconds")


# ===== MAIN ENTRY POINT =====
def main():
    """Run the MVP system."""
    config = Config()
    system = AgenticSystem(config)

    # Run refinement with default or custom prompt
    initial_prompt = input(
        "Enter initial prompt (or press Enter for default): "
    ).strip()
    if not initial_prompt:
        initial_prompt = "Write a compelling news article about food insecurity that includes real human stories"

    system.run_refinement_cycle(initial_prompt)


if __name__ == "__main__":
    main()
