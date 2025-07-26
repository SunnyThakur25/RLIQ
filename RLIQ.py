# 🚀 RLIQ 3.0: Dynamic Dual-Mode AI with Circle of Thought, Scoring & Visualization
# Truth-Seeking Mode: Socratic + Paradoxical | Simple Mode: Chain of Thought
# Everything is dynamic — no hardcoded logic

from typing import Dict, Optional, List, Any
import torch
import random
import logging
import json
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# === CONFIGURATION ===
@dataclass
class RLIQConfig:
    MODEL_NAME: str = "Qwen/Qwen-14B-Chat"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    QUANTIZATION: bool = True
    WARMUP_QUESTIONS: List[str] = None

    def __post_init__(self):
        if self.WARMUP_QUESTIONS is None:
            self.WARMUP_QUESTIONS = [
                "What's the most important question you've never asked?",
                "When do you feel most alive in your thinking?",
                "How do you prefer to explore new ideas?"
            ]

    MODES = {
        "TRUTH": {
            "name": "🌀 Truth-Seeking Mode",
            "temperature": 0.95,
            "max_tokens": 512,
            "system_prompt": (
                "You are a truth-seeking AI guide. Your role is to help the user think deeply, "
                "question assumptions, and explore multiple perspectives. Never give final answers. "
                "Ask one profound question at a time. Use paradox only when needed. "
                "Be adaptive: if the user repeats, go deeper. If they resist, soften. "
                "Never say 'I am an AI'. Be a mirror of awareness."
            )
        },
        "SIMPLE": {
            "name": "✨ Simple Answer Mode",
            "temperature": 0.6,
            "max_tokens": 256,
            "system_prompt": (
                "You are a helpful assistant. Provide clear, step-by-step reasoning. "
                "Use Chain of Thought: Think step by step. Be practical and efficient."
            )
        }
    }


# === MODE MANAGER ===
class ModeManager:
    """Manages persistent user mode and conversation state"""
    
    def __init__(self):
        self.user_profiles: Dict[str, Dict] = {}

    def initialize_user(self, user_id: str):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "mode": None,
                "history": [],  # [{"input": str, "response": str, "score": dict}]
                "conversation_flow": [],  # For visualization
                "last_prompt": ""
            }

    def get_mode(self, user_id: str) -> str:
        self.initialize_user(user_id)
        return self.user_profiles[user_id]["mode"]

    def set_mode(self, user_id: str, mode: str):
        self.initialize_user(user_id)
        self.user_profiles[user_id]["mode"] = mode


# === DYNAMIC QUESTION ENGINE  ===
class DynamicQuestionEngine:
    """Generates adaptive, context-aware questions using the model itself"""
    
    def __init__(self, tokenizer, model, device="cuda"):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def generate(self, user_input: str, history: List[str], mode: str) -> str:
        if mode == "SIMPLE":
            return self._chain_of_thought(user_input)
        else:
            return self._socratic_question(user_input, history)

    def _chain_of_thought(self, prompt: str) -> str:
        return f"{prompt}\n\nLet's think step by step:\n"

    def _socratic_question(self, user_input: str, history: List[str]) -> str:
        context = "\n".join(history[-4:])  # Last 4 exchanges
        prompt = f"""
You are a Socratic Guide. Your role is to help the user think deeply by asking one profound question at a time.
Never give answers. Only ask questions that reveal assumptions, challenge beliefs, or open new perspectives.
Follow a circle of thought pattern backward and forward reasoning.

Conversation so far:
{context}

Human: {user_input}
AI:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.9,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract AI response
        if "AI:" in response:
            question = response.split("AI:")[-1].strip()
        else:
            question = response.strip()
        
        # Clean up
        if "?" in question:
            question = question.split("?")[0] + "?"
        else:
            question = question[:100] + "?"  # Fallback
        
        return f"AI: {question}"


# === RESPONSE SCORER ===
class ResponseScorer:
    """Scores AI responses on truth-seeking metrics"""
    
    def score(self, response: str) -> Dict[str, int]:
        return {
            "Curiosity": 1 if any(w in response.lower() for w in ["why", "how", "what if"]) else 0,
            "Humility": 1 if any(w in response.lower() for w in ["don't know", "uncertain", "maybe"]) else 0,
            "Depth": len(response.split()) // 50,  # Rough depth
            "Paradox": 1 if "—" in response or "yet" in response.lower() else 0,
            "OpenEnded": 1 if response.strip().endswith("?") else 0
        }


# === VISUALIZER ===
class ConversationVisualizer:
    """Visualizes conversation flow and scoring"""
    
    def plot_circle_of_thought(self, flow: List[Dict]):
        scores = [sum(d['score'].values()) for d in flow]
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(scores)+1), scores, 'o-', color='purple')
        plt.title("Circle of Thought: Depth Over Time")
        plt.xlabel("Turn")
        plt.ylabel("Inquiry Score")
        plt.grid(True)
        plt.show()

    def show_conversation_html(self, flow: List[Dict]):
        html = "<div style='font-family: Arial; line-height: 1.6;'>"
        for i, turn in enumerate(flow):
            role = "👤 You" if i % 2 == 0 else "🤖 RLIQ"
            msg = turn.get("input", "") if "input" in turn else turn.get("response", "")
            html += f"<p><b>{role}:</b> {msg}</p>"
        html += "</div>"
        display(HTML(html))


# === DUAL-MODE PROCESSOR ===
class DualModeProcessor:
    def __init__(self, config: RLIQConfig):
        self.config = config
        self.mode_manager = ModeManager()
        self.tokenizer = None
        self.model = None
        self.question_engine = None
        self.scorer = ResponseScorer()
        self.visualizer = ConversationVisualizer()
        self._load_model()

    def _load_model(self):
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
                
            
            ) if self.config.QUANTIZATION else None

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.MODEL_NAME,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quant_config,
                trust_remote_code=True
                
            )
            
            self.question_engine = DynamicQuestionEngine(self.tokenizer, self.model, self.config.DEVICE)
            logging.info("Model and engines loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def _build_prompt(self, text: str, mode: str) -> str:
        config = RLIQConfig.MODES[mode]
        system_prompt = config["system_prompt"]
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.DEVICE)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "assistant" in full_text:
            return full_text.split("assistant")[-1].split("<|im_end|>")[0].strip()
        return full_text.strip()

    def process(self, text: str, user_id: str) -> Dict[str, Any]:
        self.mode_manager.initialize_user(user_id)
        profile = self.mode_manager.user_profiles[user_id]
        history = [turn["input"] for turn in profile["history"]]

        # --- DYNAMIC MODE SWITCHING ---
        user_text_lower = text.strip().lower()
        if user_text_lower in ["truth", "go to truth"]:
            self.mode_manager.set_mode(user_id, "TRUTH")
            response = "🌀 Switched to **Truth-Seeking Mode**.\n\nAsk a question — let’s explore it deeply."
            return {"response": response, "mode_used": "TRUTH"}

        elif user_text_lower in ["simple", "go to simple"]:
            self.mode_manager.set_mode(user_id, "SIMPLE")
            response = "✨ Switched to **Simple Answer Mode**.\n\nAsk anything — I’ll explain step by step."
            return {"response": response, "mode_used": "SIMPLE"}

        # Use current mode
        mode = self.mode_manager.get_mode(user_id)
        if not mode:
            mode = "TRUTH"  # Default
            self.mode_manager.set_mode(user_id, mode)
            greeting = f"✅ Mode set to: **{RLIQConfig.MODES[mode]['name']}**\n\n"
        else:
            greeting = ""

        # Generate dynamic prompt
        if mode == "SIMPLE":
            enhanced_text = self.question_engine._chain_of_thought(text)
            full_prompt = self._build_prompt(enhanced_text, mode)
        else:
            question = self.question_engine._socratic_question(text, history)
            full_prompt = self._build_prompt(text, mode)

        # Generate response
        raw_response = self._generate(full_prompt, RLIQConfig.MODES[mode]["max_tokens"], RLIQConfig.MODES[mode]["temperature"])
        
        # Post-process
        if mode == "TRUTH":
            final_response = f"{question}\n\n{raw_response}"
            final_response += (
                "\n\n💡 What if the opposite is also true?\n"
                "🔄 Reply 'simple' to switch modes, or ask your next question."
            )
        else:
            final_response = raw_response + (
                "\n\n🌀 Reply 'truth' to explore this philosophically, "
                "or 'simple' to stay practical."
            )

        # Score response
        score = self.scorer.score(final_response)

        # Log
        profile["history"].append({
            "input": text,
            "response": final_response,
            "score": score,
            "mode": mode
        })
        profile["conversation_flow"].append({
            "turn": len(profile["history"]),
            "input": text,
            "response": final_response,
            "score": score
        })

        return {
            "response": greeting + final_response,
            "mode_used": mode,
            "score": score,
            "flow": profile["conversation_flow"]
        }


# === INTERACTIVE INTERFACE ===
class InteractiveInterface:
    def __init__(self):
        self.config = RLIQConfig()
        self.processor = DualModeProcessor(self.config)
        self.mode_manager = self.processor.mode_manager

    def start_conversation(self, user_id: str):
        self.mode_manager.initialize_user(user_id)
        print(f"\n🤖 RLIQ 3.0: Hello {user_id}!")
        print("I’ll adapt to your thinking style.")
        print("💡 Just ask your first question.")

        while True:
            try:
                user_input = input(f"\n👤 You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "bye"]:
                    break
                
                result = self.processor.process(user_input, user_id)
                print(f"\n🤖 {result['response']}")
                
                # After 3+ turns, offer visualization
                if len(result["flow"]) >= 3:
                    print("\n📊 Type 'visualize' to see your thinking journey.")
                    if input(">").strip().lower() == "visualize":
                        self.processor.visualizer.plot_circle_of_thought(result["flow"])
                        self.processor.visualizer.show_conversation_html(result["flow"])
                        break
                        
            except KeyboardInterrupt:
                print("\n\n👋 Conversation ended. Thank you for thinking deeply.")
                break
            except Exception as e:
                print(f"\n⚠️  Error: {e}")

    def show_analytics(self):
        print("\n📊 RLIQ Usage Analytics")
        print("-" * 40)
        for user_id, profile in self.mode_manager.user_profiles.items():
            mode = profile["mode"] or "Unset"
            count = len(profile["history"])
            avg_score = sum(sum(t.get("score", {}).values()) for t in profile["history"]) / max(1, count)
            print(f"User: {user_id} | Mode: {mode} | Turns: {count} | Avg Score: {avg_score:.2f}")


# === DEMO & ENTRY POINT ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    interface = InteractiveInterface()
    
    print("🚀 RLIQ 3.0: Dynamic Truth-Seeking AI")
    print("A living, adaptive intelligence — no hardcoded logic")
    print("=" * 60)
    
    while True:
        user_id = input("\nEnter User ID (or 'exit' to quit): ").strip()
        if user_id.lower() == "exit":
            break
        if not user_id:
            continue
            
        interface.start_conversation(user_id)
        
        print(f"\n💬 Session Complete for {user_id}")
        print("1. New Conversation")
        print("2. Show Analytics")
        print("3. Exit")
        choice = input("Choose: ").strip()
        if choice == "2":
            interface.show_analytics()
        elif choice == "3":
            break